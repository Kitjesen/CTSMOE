"""One serialized 958-update stair-coverage experiment, then stop for review."""
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

from runtime_contract import validate_coverage, verify_prepared

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V20 = ROOT / 'speed_brake_v20'
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
AUDIT = V20 / 'landing_audit_v21'
OWNER = False
TARGET = 34823


def budget(completed):
    spent = completed - 31831 + 8
    if not 2042 <= spent <= 3000 or not 33865 <= completed <= TARGET:
        raise ValueError('Outside the authorized V21 first-stage budget')
    return dict(authorized_new_updates=10000, spent_new_updates=spent,
                remaining_new_updates=10000-spent, discarded_optimizer_updates=8,
                v21_new_updates=completed-33865, v21_limit=958)


def status(state, completed, **extra):
    item = dict(state=state, supervisor_pid=os.getpid(), completed_updates=completed,
                stage_target=TARGET, time_utc=datetime.now(timezone.utc).isoformat(),
                automatic_extension=False, **budget(completed), **extra)
    tmp = HERE / f'pipeline_status.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(item, indent=2))
    tmp.replace(HERE / 'pipeline_status.json')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def verify_code():
    verify_prepared(HERE)
    for folder, manifest in [(HERE/'code', HERE/'PREPARED.json'),
                             (HERE, HERE/'ORCHESTRATION_MANIFEST.json'),
                             (V20/'code', V20/'PREPARED.json')]:
        for name, digest in json.loads(manifest.read_text())['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Frozen file changed: {folder/name}')
    for path, digest in json.loads((HERE/'EXTERNAL_MANIFEST.json').read_text())['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen evaluation dependency changed: {path}')


def run(stage, argv, completed):
    with (HERE/f'{stage}.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        status(stage, completed, child_pid=child.pid)
        rc = child.wait()
    if rc:
        raise RuntimeError(f'{stage} failed ({rc}); preserve evidence and used budget')


def checkpoint_identity(source, completed):
    import torch
    saved = torch.load(source, map_location='cpu', weights_only=False)
    if saved.get('completed_updates') != completed or saved.get('common_step_counter') != completed*24:
        raise RuntimeError('Wrong checkpoint update or simulation-step count')
    for key in ('model_state_dict', 'optimizer_state_dict', 'optimizer_stu_enc_state_dict'):
        if not saved.get(key):
            raise RuntimeError(f'Missing checkpoint state: {key}')
    return dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(), completed=completed,
                common_step_counter=saved['common_step_counter'])


def check_resume(output, source, completed, increment, identity):
    row = json.loads((output/'run.json').read_text())
    if (row['resume'] != str(source) or row['starting_completed_updates'] != completed
            or row['additional_updates'] != increment
            or row['target_completed_updates'] != completed+increment
            or row['resume_sha256'] != identity['sha256']
            or identity['completed'] != completed or identity['common_step_counter'] != completed*24
            or hashlib.sha256(source.read_bytes()).hexdigest() != identity['sha256']):
        raise RuntimeError('Wrong training source or update budget')
    required = ('model', 'ppo_optimizer', 'student_optimizer', 'terrain_assignment',
                'terrain_restored_exactly', 'command_terrain_membership',
                'terrain_allocation', 'stair_episode_totals')
    if not all(row['resume_checks'].get(k) is True for k in required):
        raise RuntimeError('Incomplete optimizer/terrain restoration')
    coverage = json.loads((output/'COVERAGE_GEOMETRY.json').read_text())
    validate_coverage(coverage)
    if row['terrain_allocation']['pending_moves'] != 0:
        raise RuntimeError('Unexpected pending allocation migration')
    return row


def main():
    global OWNER
    completed = 33865
    with (ROOT/'pipeline.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE/'STARTED.json').exists() or (HERE/'pipeline_status.json').exists():
            raise RuntimeError('Existing V21 attempt; do not duplicate or overwrite')
        verify_code()
        identity = json.loads((HERE/'SOURCE_IDENTITY.json').read_text())
        source = Path(identity['path'])
        if (identity['completed_updates'] != completed
                or hashlib.sha256(source.read_bytes()).hexdigest() != identity['sha256']):
            raise RuntimeError('Source33865 identity mismatch')
        ready = json.loads((HERE/'VERIFIED_READY.json').read_text())
        if (ready['state'] != 'verified_not_training' or ready['source_completed'] != completed
                or ready['source_sha256'] != identity['sha256']
                or ready.get('actual_runner_resume_verified') is not True):
            raise RuntimeError('Requires independently verified zero-update preflight')
        audit = json.loads((AUDIT/'evaluation_33865_corrected/FINALIZED.json').read_text())
        if audit != dict(state='complete', cases=96, optimizer_updates=0, hashes_unchanged=True,
                         source_completed=33174, candidate_completed=33865):
            raise RuntimeError('Incomplete corrected baseline diagnostic')
        shown = json.loads((HERE/'SHOWCASE_PRESENTED.json').read_text())
        if shown.get('completed_updates') != completed or not shown.get('presented_to_user'):
            raise RuntimeError('User requested baseline showcase before training')
        original = module('original_pipeline', ROOT/'pipeline.py')
        v20_driver = module('v20_driver', V20/'driver_v2.py')
        qualified = module('v21_qualified_command', HERE/'run_preflight.py')
        previous = json.loads((V20/'requested_evaluation_33865/VERDICT.json').read_text())
        if previous['stop_for_review']:
            raise RuntimeError('A behavior rejection must not be bypassed')
        with (HERE/'STARTED.json').open('x') as f:
            json.dump(dict(pid=os.getpid(), source=identity, **budget(completed)), f, indent=2)
        OWNER = True
        status('starting_qualified_training', completed)
        for increment in (16, 942):
            target = completed+increment
            budget(target)
            output = HERE/f'training_{completed}_to_{target}'
            with (BASE/'full_episode_v12_pipeline.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                verify_code()
                if shutil.disk_usage(ROOT).free < 10*1024**3:
                    raise RuntimeError('Less than 10 GiB free')
                if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True).strip():
                    raise RuntimeError('GPU occupied')
                if output.exists():
                    raise RuntimeError('Training output already exists')
                source_identity = checkpoint_identity(source, completed)
                args = qualified.command(HERE, {'path': str(source)})
                args[2] = str(HERE/'code/train.py')
                for key, value in (('--resume', source), ('--output', output), ('--updates', increment)):
                    args[args.index(key)+1] = str(value)
                with (HERE/f'launch_{target}.json').open('x') as f:
                    json.dump(dict(args=args, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                   **budget(completed)), f, indent=2)
                run(f'train_{target}', args, completed)
                result = json.loads((output/'SUCCESS.json').read_text())
                if result.get('completed_updates') != target:
                    raise RuntimeError('Unexpected completed count; audit before recovery')
                check_resume(output, source, completed, increment, source_identity)
                checkpoint_identity(output/f'model_{target-1}.pt', target)
            source, completed = output/f'model_{target-1}.pt', target
            status('training_segment_complete', completed, checkpoint=str(source))
        evaluation = HERE/f'evaluation_{completed}'
        run('paired', [sys.executable, str(ROOT/'evaluate_pair.py'), '--output', str(evaluation),
                      '--sources', f'source31831:31831:{original.SOURCE}',
                      f'candidate:{completed}:{source}'], completed)
        run('braking', [sys.executable, str(HERE/'run_braking.py'), '--output', str(HERE/'braking_34823'),
                       '--candidate', str(source), '--completed', str(completed)], completed)
        run('stair_terminal', [sys.executable, str(AUDIT/'run_audit.py'),
                              '--source', identity['path'], '--source-completed', '33865',
                              '--candidate', str(source), '--completed', str(completed),
                              '--output', str(HERE/'stair_terminal_34823')], completed)
        verdict = v20_driver.verdict_for(original, evaluation, previous, completed)
        verdict['automatic_continuation'] = False
        verdict['next_review'] = 'Joint review of posture, speed, qualified braking, all terrains and local terminal traces'
        with (evaluation/'CONTINUATION.json').open('x') as f:
            json.dump(verdict, f, indent=2)
        verify_code()
        status('behavior_rejected' if verdict['stop_for_review'] else 'complete_pending_review',
               completed, checkpoint=str(source), evaluation=str(evaluation))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        if OWNER:
            prior = json.loads((HERE/'pipeline_status.json').read_text())
            status('technical_failure', prior['completed_updates'], error=str(exc),
                   last_stage=prior, automatic_retry=False)
        raise
