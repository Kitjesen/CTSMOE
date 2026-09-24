"""User-requested zero-update evaluation after an atomic checkpointed stop."""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAINING = HERE / 'training_33174_to_34823'
OUT = HERE / 'requested_evaluation_33865'
CANDIDATE = TRAINING / 'model_33864.pt'
COMPLETED = 33865
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def status(state, **extra):
    data = dict(state=state, pid=os.getpid(), completed_updates=COMPLETED,
                optimizer_updates=0, automatic_training=False,
                time_utc=datetime.now(timezone.utc).isoformat(), **extra)
    temp = OUT / f'status.{os.getpid()}.tmp'
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(OUT / 'status.json')


def run(stage, command):
    with (OUT / f'{stage}.log').open('x') as log:
        child = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        status(stage, child_pid=child.pid)
        rc = child.wait()
    if rc:
        raise RuntimeError(f'{stage} failed ({rc}); preserve existing evidence')


def main():
    with (ROOT / 'pipeline.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped = json.loads((TRAINING / 'status.json').read_text())
        if stopped != dict(state='stopped', error='STOP requested after saving'):
            raise RuntimeError('Requires the requested checkpointed stop')
        if not (TRAINING / 'STOP').exists() or not (HERE / 'USER_EVALUATION_REQUEST.json').exists():
            raise RuntimeError('Missing explicit evaluation transition record')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU still occupied')
        driver = load_module('v20_verified_driver', HERE / 'driver_v2.py')
        driver.verify_code()
        original = load_module('v19_original_pipeline', ROOT / 'pipeline.py')
        import torch
        checkpoint = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
        if checkpoint['completed_updates'] != COMPLETED:
            raise RuntimeError('Unexpected completed update count')
        for key in ('model_state_dict', 'optimizer_state_dict', 'optimizer_stu_enc_state_dict'):
            if not checkpoint.get(key):
                raise RuntimeError(f'Missing checkpoint state: {key}')
        identity = dict(path=str(CANDIDATE), completed_updates=COMPLETED,
                        sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                        bytes=CANDIDATE.stat().st_size,
                        common_step_counter=checkpoint['common_step_counter'],
                        **driver.budget(COMPLETED))
        del checkpoint
        OUT.mkdir(exist_ok=False)
        (OUT / 'SOURCE_IDENTITY.json').write_text(json.dumps(identity, indent=2))
        status('starting')
        pair = OUT / 'paired'
        run('paired', [sys.executable, str(ROOT / 'evaluate_pair.py'), '--output', str(pair),
                       '--sources', f'source31831:31831:{original.SOURCE}',
                       f'candidate:{COMPLETED}:{CANDIDATE}'])
        run('focused', [sys.executable, str(HERE / 'run_focus.py'), '--output', str(OUT / 'focused'),
                        '--candidate', str(CANDIDATE), '--completed', str(COMPLETED)])
        previous = json.loads((ROOT / 'evaluation_32831/CONTINUATION.json').read_text())
        verdict = driver.verdict_for(original, pair, previous, COMPLETED)
        verdict['automatic_training'] = False
        verdict['allocation_change'] = 'not_applied_pending_user_requested_review'
        (OUT / 'VERDICT.json').write_text(json.dumps(verdict, indent=2))
        focus = json.loads((OUT / 'focused/FINALIZED.json').read_text())
        if focus != dict(state='complete', optimizer_updates=0, hashes_unchanged=True,
                         smoke_only=False, candidate_completed=COMPLETED):
            raise RuntimeError('Focused assessment incomplete')
        if hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() != identity['sha256']:
            raise RuntimeError('Candidate changed during evaluation')
        driver.verify_code()
        (OUT / 'FINALIZED.json').write_text(json.dumps(dict(
            state='complete_pending_behavior_review', optimizer_updates=0,
            candidate_completed=COMPLETED, hashes_unchanged=True,
            original_cases=412, braking_cases=144, stair_terminal_cases=96,
            automatic_training=False, **driver.budget(COMPLETED)), indent=2))
        status('complete_pending_behavior_review', verdict=str(OUT / 'VERDICT.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        if OUT.exists():
            status('technical_failure', error=str(exc), automatic_retry=False)
        raise
