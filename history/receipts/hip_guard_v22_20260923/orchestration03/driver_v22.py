"""V22: qualified hip-envelope repair, 16+984 updates, then paired tests and stop."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from runtime_contract import BASE, COMPLETED, verify_prepared
from run_preflight import command

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET = 35823
RESTORE = ('model','ppo_optimizer','student_optimizer','terrain_assignment',
           'command_terrain_membership','terrain_restored_exactly','terrain_allocation','stair_episode_totals')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_all():
    identity, _ = verify_prepared(HERE)
    manifest = json.loads((HERE/'ORCHESTRATION_V22.json').read_text())
    for name, digest in manifest['files'].items():
        if file_hash(HERE/name) != digest:
            raise RuntimeError(f'Frozen V22 orchestration changed: {name}')
    for name, digest in manifest['external'].items():
        if file_hash(name) != digest:
            raise RuntimeError(f'Frozen evaluation dependency changed: {name}')
    return identity


def status(stage, completed, **more):
    if not COMPLETED <= completed <= TARGET:
        raise ValueError('V22 update budget exceeded')
    value = dict(stage=stage, completed_updates=completed, target_completed_updates=TARGET,
        added_updates=completed-COMPLETED, spent_new_updates=completed-31831+8,
        remaining_new_updates=10000-(completed-31831+8), automatic_extension=False,
        pid=os.getpid(), time_utc=datetime.now(timezone.utc).isoformat(), **more)
    p = HERE/'pipeline_status.tmp'
    p.write_text(json.dumps(value,indent=2))
    p.replace(HERE/'pipeline_status.json')


def execute(stage, argv, completed):
    verify_all()
    with (HERE/f'{stage}.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        status(stage,completed,child_pid=child.pid)
        result = child.wait()
    if result:
        raise RuntimeError(f'{stage} exited {result}; preserve evidence, no automatic retry')
    verify_all()


def check_checkpoint(path, completed):
    import torch
    saved = torch.load(path,map_location='cpu',weights_only=False)
    if saved['completed_updates'] != completed or saved['common_step_counter'] != completed*24:
        raise ValueError('Wrong checkpoint age')
    if completed > COMPLETED and saved.get('hip_guard_profile') != 'hip_inward012_v22':
        raise ValueError('New checkpoint omitted hip contract')
    if not all(saved.get(k) for k in ('model_state_dict','optimizer_state_dict','optimizer_stu_enc_state_dict')):
        raise ValueError('Missing training states')


def main():
    completed = COMPLETED
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'STARTED_V22.json').exists():
            raise RuntimeError('V22 was already started; no duplicate runs')
        identity = verify_all()
        if not (ROOT/'hip_guard_v22_attempt02/READY_FOR_REVIEW.json').exists():
            raise RuntimeError('Requires completed zero-update preflight')
        with (HERE/'STARTED_V22.json').open('x') as f:
            json.dump(dict(source=identity, max_added_updates=1000),f,indent=2)
        try:
            execute('independent_qualification',[sys.executable,str(HERE/'verify_metadata_bridge.py')],completed)
            ready = json.loads((HERE/'VERIFIED_READY.json').read_text())
            if ready['state'] != 'verified_not_training' or ready['source_completed'] != completed:
                raise RuntimeError('Independent qualification did not pass')
            source = Path(identity['path'])
            # Establish old weights in the new operating envelope before adapting them.
            execute('baseline_assessment',[sys.executable,str(HERE/'assessment/run_assessment.py'),
                '--output',str(HERE/'baseline_34823_new_hip'),
                '--sources',f'source_new_hip:{completed}:{source}'],completed)
            for increment in (16,984):
                target = completed+increment
                output = HERE/f'training_{completed}_to_{target}'
                with (BASE/'full_episode_v12_pipeline.lock').open('a') as gpu_lock:
                    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                        raise RuntimeError('GPU occupied')
                    if shutil.disk_usage(HERE).free < 15*1024**3 or output.exists():
                        raise RuntimeError('Insufficient disk or existing output; no overwrite')
                    check_checkpoint(source,completed)
                    digest = file_hash(source)
                    argv = command(HERE,{'path':str(source)})
                    argv[2] = str(HERE/'code/train.py')
                    for key,value in (('--output',output),('--updates',increment)):
                        argv[argv.index(key)+1] = str(value)
                    (HERE/f'launch_{target}.json').write_text(json.dumps(dict(args=argv, source_sha256=digest,
                        target=target, new_updates=increment),indent=2))
                    execute(f'train_{target}',argv,completed)
                    receipt = json.loads((output/'run.json').read_text())
                    if (receipt['resume_sha256'] != digest or file_hash(source) != digest
                            or not all(receipt['resume_checks'].get(k) is True for k in RESTORE)
                            or receipt['hip_guard_profile'] != 'hip_inward012_v22'
                            or receipt['additional_updates'] != increment
                            or receipt['target_completed_updates'] != target):
                        raise RuntimeError('Source/optimizer/difficulty/budget restore mismatch')
                    result = json.loads((output/'SUCCESS.json').read_text())
                    if result['completed_updates'] != target:
                        raise RuntimeError('Unexpected final completed count')
                    source = output/f'model_{target-1}.pt'
                    check_checkpoint(source,target)
                    completed = target
            execute('final_paired_assessment',[sys.executable,str(HERE/'assessment/run_assessment.py'),
                '--output',str(HERE/'evaluation_35823'), '--sources',
                f'source_new_hip:{COMPLETED}:{identity["path"]}',f'candidate:{completed}:{source}'],completed)
            status('complete_pending_behavior_review',completed,checkpoint=str(source),training_running=False)
        except Exception as exc:
            # A mid-segment failure may have performed updates after the last save.
            # Record the highest observed iteration; do not silently reclaim budget.
            observed = completed
            for output in HERE.glob('training_*'):
                metrics = output/'metrics.jsonl'
                if metrics.exists():
                    for line in metrics.read_text().splitlines():
                        observed = max(observed,int(json.loads(line)['completed_updates']))
            status('technical_failure',observed,error=str(exc),training_running=False,automatic_retry=False)
            raise


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
