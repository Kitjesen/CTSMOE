"""Resume completed recovery qualification after correcting the verifier source path."""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
SOURCE = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922/recovery01/training_32831_to_34823/model_33173.pt')
SOURCE_SHA = '05c34f02dcfb0bec462640c361258a896a3435e0684f3c3b8ae15a6175b0ef83'
LOST_UPDATES = 8
REMAINING_STAGES = (1649, 3000, 4000)
OWNER = False


def budget(completed):
    retained = completed - 31831
    spent = retained + LOST_UPDATES
    if not 1008 <= spent <= 10000:
        raise ValueError('Recovery exceeds authorized optimizer budget')
    return dict(retained_new_updates=retained, discarded_optimizer_updates=LOST_UPDATES,
                spent_new_updates=spent, remaining_new_updates=10000-spent)


def write_status(state, completed, **fields):
    row = dict(state=state, supervisor_pid=os.getpid(), time_utc=datetime.now(timezone.utc).isoformat(),
               source_completed=31831, original_target_completed=41831, target_completed=41823,
               authorized_new_updates=10000, completed_updates=completed, recovery='speed_brake_v20',
               **budget(completed), **fields)
    p = HERE/f'pipeline_status.{os.getpid()}.tmp'
    p.write_text(json.dumps(row, indent=2))
    p.replace(HERE/'pipeline_status.json')


def verify_code():
    for folder, manifest in [(ROOT/'code', ROOT/'PREPARED.json'),
                             (ROOT/'recovery01/code', ROOT/'recovery01/PREPARED.json'),
                             (HERE/'code', HERE/'PREPARED.json'),
                             (HERE, HERE/'ORCHESTRATION_MANIFEST_v2.json')]:
        for name, expected in json.loads(manifest.read_text())['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen runtime changed: {folder/name}')


def run(stage, argv, completed):
    with (HERE/f'{stage}.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=HERE/'code', stdout=log, stderr=subprocess.STDOUT)
        write_status(stage, completed, child_pid=child.pid)
        rc = child.wait()
    if rc:
        raise RuntimeError(f'{stage} failed ({rc}); preserve evidence; no automatic retry')


def verdict_for(original, evaluation, previous, completed):
    final = json.loads((evaluation/'FINALIZED.json').read_text())
    if final != dict(state='complete', optimizer_updates=0, hashes_unchanged=True, terrain_cases=332, flat_cases=80):
        raise RuntimeError('Incomplete paired evaluation')
    verdict = original.collapse_metrics(evaluation)
    repeated = sorted(set(verdict['terrain_regressions']) & set(previous['terrain_regressions']))
    repeated_pose = sorted(set(verdict['pose_regressions']) & set(previous['pose_regressions']))
    no_lift = bool(previous['no_candidate_lift'] and verdict['no_candidate_lift'])
    # Original +3000/+6000/+10000 gates include the eight updates spent before
    # the logging fault. They cannot silently become additional training credit.
    stop = verdict['severe'] or (budget(completed)['spent_new_updates'] >= 3000
                               and bool(repeated or no_lift or repeated_pose))
    verdict.update(repeated_terrain_regressions=repeated, repeated_pose_regressions=repeated_pose,
                   repeated_no_lift=no_lift, stop_for_review=stop, **budget(completed))
    return verdict


def main():
    global OWNER
    completed = 33174
    with (ROOT/'pipeline.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE/'STARTED_V20.json').exists() or (HERE/'pipeline_status.json').exists():
            raise RuntimeError('V20 already attempted; preserve state instead of duplicate launch')
        old_output = SOURCE.parent
        stopped = json.loads((old_output/'status.json').read_text())
        if stopped != {'state': 'stopped', 'error': 'STOP requested after saving'} or not (old_output/'STOP').exists():
            raise RuntimeError('Requires exact checkpointed user-requested V19 stop')
        verify_code()
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
            raise RuntimeError('Checkpoint33174 changed')
        OWNER = True
        with (HERE/'STARTED_V20.json').open('x') as f:
            json.dump(dict(pid=os.getpid(), stopped_source=stopped, **budget(completed)), f, indent=2)
        write_status('starting_qualification', completed)
        spec = importlib.util.spec_from_file_location('v19_original_pipeline', ROOT/'pipeline.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        previous = json.loads((ROOT/'evaluation_32831/CONTINUATION.json').read_text())
        if previous['stop_for_review']:
            raise RuntimeError('Behavior rejection cannot be bypassed')
        run('preflight_driver', [sys.executable, str(HERE/'run_preflight.py')], completed)
        run('verify_driver', [sys.executable, str(HERE/'verify_preflight_v2.py')], completed)
        ready = json.loads((HERE/'VERIFIED_READY.json').read_text())
        if (ready.get('state') != 'verified_not_training' or ready.get('source_completed') != completed
                or ready.get('source_sha256') != SOURCE_SHA
                or ready.get('actual_runner_resume_verified') is not True):
            raise RuntimeError('Incomplete independently verified V20 qualification')
        run('focused_smoke', [sys.executable, str(HERE/'run_focus.py'), '--smoke',
            '--output', str(HERE/'focused_smoke'),
            '--candidate', str(ROOT/'training_31831_to_32831/model_32830.pt'), '--completed', '32831'], completed)
        source = SOURCE
        for increment in REMAINING_STAGES:
            target = completed + increment
            budget(target)
            output = HERE/f'training_{completed}_to_{target}'
            with (BASE/'full_episode_v12_pipeline.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                verify_code()
                if shutil.disk_usage(ROOT).free < 10*1024**3:
                    raise RuntimeError('Less than 10GiB free')
                if output.exists():
                    raise RuntimeError('Training output already exists')
                if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
                    raise RuntimeError('GPU occupied')
                args = original.qualified_args()
                args[2] = str(HERE/'code/train.py')
                for key, value in (('--resume', source), ('--output', output), ('--updates', increment)):
                    args[args.index(key)+1] = str(value)
                with (HERE/f'launch_{target}.json').open('x') as f:
                    json.dump(dict(args=args, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                   **budget(completed)), f, indent=2)
                run(f'train_{target}', args, completed)
                result = json.loads((output/'SUCCESS.json').read_text())
                if result.get('completed_updates') != target:
                    raise RuntimeError('Unexpected update count')
            source, completed = output/f'model_{target-1}.pt', target
            evaluation = HERE/f'evaluation_{target}'
            run(f'evaluate_{target}', [sys.executable, str(ROOT/'evaluate_pair.py'), '--output', str(evaluation),
                '--sources', f'source31831:31831:{original.SOURCE}', f'candidate:{completed}:{source}'], completed)
            run(f'focus_{completed}', [sys.executable, str(HERE/'run_focus.py'),
                '--output', str(HERE/f'focus_{completed}'), '--candidate', str(source),
                '--completed', str(completed)], completed)
            verdict = verdict_for(original, evaluation, previous, completed)
            with (evaluation/'CONTINUATION.json').open('x') as f:
                json.dump(verdict, f, indent=2)
            if verdict['stop_for_review']:
                write_status('regression_or_no_lift_review_required', completed, checkpoint=str(source),
                             evidence=str(evaluation), automatic_continuation=False)
                return
            previous = verdict
        write_status('completed_10000_budget_pending_review', completed, checkpoint=str(source), automatic_continuation=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        if OWNER:
            current = json.loads((HERE/'pipeline_status.json').read_text())
            if current.get('supervisor_pid') == os.getpid():
                write_status('technical_failure', current['completed_updates'], error=str(exc),
                             last_stage=current, automatic_retry=False)
        raise
