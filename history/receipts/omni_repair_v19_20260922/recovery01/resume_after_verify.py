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
SOURCE = ROOT/'training_31831_to_32831/model_32830.pt'
SOURCE_SHA = 'f40891182ce1b0f9d80adac2c962b74c6ad8226e183fc44eb62cf9cba4ddd32a'
LOST_UPDATES = 8
REMAINING_STAGES = (1992, 3000, 4000)
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
               authorized_new_updates=10000, completed_updates=completed, recovery='recovery01',
               **budget(completed), **fields)
    p = ROOT/f'pipeline_status.{os.getpid()}.tmp'
    p.write_text(json.dumps(row, indent=2))
    p.replace(ROOT/'pipeline_status.json')


def verify_code():
    for folder, manifest in [(ROOT/'code', ROOT/'PREPARED.json'), (HERE/'code', HERE/'PREPARED.json')]:
        for name, expected in json.loads(manifest.read_text())['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen runtime changed: {folder/name}')
    for name, expected in json.loads((HERE/'RECOVERY_MANIFEST.json').read_text())['files'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Recovery orchestration changed: {name}')
    for name, expected in json.loads((HERE/'RESUME_VERIFY_MANIFEST.json').read_text())['files'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Corrected verifier orchestration changed: {name}')


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
    completed = 32831
    with (ROOT/'pipeline.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = json.loads((ROOT/'pipeline_status.json').read_text())
        if current['state'] != 'technical_failure' or current.get('last_stage', {}).get('state') != 'recovery_verify':
            raise RuntimeError('Requires preserved recovery_verify technical failure')
        if current.get('completed_updates') != 32831 or current.get('spent_new_updates') != 1008:
            raise RuntimeError('Recovery budget metadata changed')
        if (HERE/'STARTED_AFTER_VERIFY.json').exists():
            raise RuntimeError('Recovery has already been attempted; do not duplicate it')
        verify_code()
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
            raise RuntimeError('Checkpoint32831 changed')
        OWNER = True
        (HERE/'STARTED_AFTER_VERIFY.json').write_text(json.dumps(dict(pid=os.getpid(), prior_status=current, **budget(completed)), indent=2))
        spec = importlib.util.spec_from_file_location('v19_original_pipeline', ROOT/'pipeline.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        previous = json.loads((ROOT/'evaluation_32831/CONTINUATION.json').read_text())
        if previous['stop_for_review']:
            raise RuntimeError('Behavior rejection cannot be bypassed')
        run('recovery_verify_v2', [sys.executable, str(HERE/'verify_preflight_v2.py')], completed)
        ready = json.loads((HERE/'VERIFIED_READY.json').read_text())
        if (ready.get('state') != 'verified_not_training' or ready.get('source_completed') != 32831
                or ready.get('source_sha256') != SOURCE_SHA
                or ready.get('actual_runner_resume_verified') is not True):
            raise RuntimeError('Incomplete independently verified recovery qualification')
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
            current = json.loads((ROOT/'pipeline_status.json').read_text())
            if current.get('supervisor_pid') == os.getpid():
                write_status('technical_failure', current['completed_updates'], error=str(exc),
                             last_stage=current, automatic_retry=False)
        raise
