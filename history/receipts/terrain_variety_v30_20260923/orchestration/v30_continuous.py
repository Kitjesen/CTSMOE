"""Qualify new scene integration, then one continuous 6000-update training run."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from v30_common import LOCK, STAGE, args, digest, frozen

STATUS = Path(__file__).with_name('CONTINUOUS_STATUS.json')


def write(state, **kwargs):
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(state=state, **kwargs), indent=2))
    temp.replace(STATUS)


def main():
    if STATUS.exists():
        raise FileExistsError('Continuous pipeline already attempted')
    write('waiting_for_preflight', additional_updates=6000, target_completed=41123)
    try:
        receipt = frozen()
        finalized = STAGE / 'preflight_3000/FINALIZED.json'
        for _ in range(120):
            if finalized.exists():
                break
            time.sleep(30)
        else:
            raise TimeoutError('Preflight did not finalize in one hour')
        check = json.loads(finalized.read_text())
        if (check['optimizer_updates'] != 0 or check['steps'] != 3000 or check['envs'] != 4096
                or check['source_sha256'] != receipt['source_sha256']
                or check['model_and_dual_optimizer_before'] != check['model_and_dual_optimizer_after']
                or check['forbidden_calls'] or not check['mutation_guard_restored']
                or check['pit_active_final'] != 0 or check['variety']['reverse_envs'] == 0):
            raise RuntimeError('Preflight independent contract check failed')
        if receipt['source_completed_updates'] != 35123:
            raise ValueError('Wrong source age')
        assessor = STAGE / 'assessment/assess_v30.py'
        assessment_sha = digest(assessor)

        def evaluate(checkpoint, label):
            write('evaluating_' + label, checkpoint=str(checkpoint))
            out = STAGE / 'assessment' / label
            with out.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable, '-u', str(assessor), '--checkpoint', str(checkpoint),
                                '--output', str(out)], stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((out / 'FINALIZED.json').read_text())
            if (not result['state_hashes_unchanged'] or result['optimizer_updates'] != 0
                    or result['selected_sha256'] != digest(checkpoint) or digest(assessor) != assessment_sha):
                raise RuntimeError('Assessment identity or mutation failure')
            return result

        with LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            frozen()
            if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                raise RuntimeError('GPU occupied')
            if shutil.disk_usage(STAGE).free < 15 * 1024**3:
                raise RuntimeError('Less than 15GiB free')
            baseline = evaluate(Path(receipt['source_path']), 'source35123')
            output = STAGE / 'training_35123_to_41123'
            if output.exists():
                raise FileExistsError('Training output already exists')
            frozen()
            with (STAGE / 'orchestration/training.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-u', str(STAGE / 'code/train.py'),
                                            *args(receipt['source_path'], output, 6000)],
                                           stdout=log, stderr=subprocess.STDOUT)
                write('training', pid=process.pid, start_completed=35123, target_completed=41123,
                      additional_updates=6000, pit_envs=0, auto_stop_every_1000=False)
                if process.wait():
                    raise RuntimeError('Training failed; checkpoint-preserving manual diagnosis required')
            completed = json.loads((output / 'SUCCESS.json').read_text())
            run = json.loads((output / 'run.json').read_text())
            if (completed['completed_updates'] != 41123 or run['additional_updates'] != 6000
                    or run['resume_sha256'] != receipt['source_sha256']):
                raise RuntimeError('Update budget mismatch')
            frozen()
            final = evaluate(output / 'model_41122.pt', 'candidate41123')
            result = {'source': baseline, 'candidate': final, 'new_updates': 6000,
                      'discarded_updates_still_budgeted': 700, 'hardware_release': False}
            (STAGE / 'assessment/COMPARISON_FINAL.json').write_text(json.dumps(result, indent=2))
            write('complete_pending_review', completed_updates=41123, further_updates=False)
    except Exception as exc:
        write('technical_failure', error=str(exc))
        raise


if __name__ == '__main__':
    main()
