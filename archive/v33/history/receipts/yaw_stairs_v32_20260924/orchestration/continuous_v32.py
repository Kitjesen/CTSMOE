"""One authorized 10000-update continuation after a completed zero-update preflight."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from v31_common import STAGE, LOCK, args, frozen, digest

STATUS = Path(__file__).with_name('STATUS_v32.json')


def status(state, **kw):
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(state=state, **kw), indent=2))
    temp.replace(STATUS)


def execute(argv, log):
    with log.open('x') as stream:
        subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    if STATUS.exists():
        raise FileExistsError('Pipeline already attempted; inspect preserved state')
    receipt = frozen()
    source = Path(receipt['source_path'])
    start = receipt['source_completed_updates']
    updates = 46194 - start
    target = start + updates
    try:
        status('preflight', start=start, target=target, new_updates=updates)
        # The original coordinator was stopped at ZERO optimizer updates after
        # detecting an inherited 10000 budget. Preserve its running preflight.
        deadline = time.monotonic() + 1800
        while Path('/proc/169418').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Existing zero-update preflight did not finish')
            time.sleep(2)
        if not (STAGE/'preflight_3000/FINALIZED.json').exists():
            raise RuntimeError('Preserved preflight failed; do not train')
        if target != 46194 or updates != 7032:
            raise RuntimeError('Authorized remaining budget mismatch')
        result = json.loads((STAGE/'preflight_3000/FINALIZED.json').read_text())
        if (result['optimizer_updates'] != 0 or result['steps'] != 3000
                or result['source_sha256'] != digest(source)
                or result['model_and_dual_optimizer_before'] != result['model_and_dual_optimizer_after']
                or result['new_command_samples'][1] != 0
                or min(result['new_command_samples'][i] for i in (0,2,3,4)) <= 0
                or min(result['yaw_reset_bins']) <= 0
                or result['landing']['force_cost_sum'] <= 0 or result['landing']['speed_cost_sum'] <= 0):
            raise RuntimeError('Independent preflight receipt verification failed')
        with LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for _ in range(30):
                gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
                if not gpu.strip():
                    break
                time.sleep(2)
            else:
                raise RuntimeError('GPU occupied after preflight process exit')
            if shutil.disk_usage(STAGE).free < 20*1024**3:
                raise RuntimeError('Less than 20 GiB free')
            assessor = STAGE/'assessment/assess_v31.py'
            def evaluate(checkpoint, name):
                status('evaluating_'+name, checkpoint=str(checkpoint), target=target)
                output = STAGE/'assessment'/name
                execute([sys.executable, '-u', str(assessor), '--checkpoint', str(checkpoint), '--output', str(output)], output.with_suffix('.log'))
                check = json.loads((output/'FINALIZED.json').read_text())
                if check['optimizer_updates'] != 0 or not check['state_hashes_unchanged']:
                    raise RuntimeError('Evaluation modified the source')
                return check
            baseline = evaluate(source, 'source')
            frozen()
            output = STAGE/f'training_{start}_to_{target}'
            if output.exists():
                raise FileExistsError(output)
            with (STAGE/'orchestration/training.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-u', str(STAGE/'code/train.py'), *args(source, output, updates)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                status('training', pid=process.pid, start=start, target=target, new_updates=updates, periodic_stop=False)
                if process.wait():
                    raise RuntimeError('Training interrupted; preserve checkpoint and diagnose')
            success = json.loads((output/'SUCCESS.json').read_text())
            if success['completed_updates'] != target:
                raise RuntimeError('Training update budget mismatch')
            candidate = evaluate(output/f'model_{target-1}.pt', 'candidate')
            (STAGE/'assessment/COMPARISON.json').write_text(json.dumps(dict(source=baseline, candidate=candidate), indent=2))
            status('complete_pending_review', completed=target, hardware_release=False)
    except Exception as exc:
        status('technical_failure', error=str(exc), start=start, target=target)
        raise


if __name__ == '__main__':
    main()
