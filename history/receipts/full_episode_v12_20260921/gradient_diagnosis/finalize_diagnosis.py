"""Independently finalize receipts after Isaac exits its process on shutdown."""
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
OUTPUT = ROOT / 'recording/full_episode_v12_gradient_diagnosis_20260921_r1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    status = json.loads((OUTPUT / 'audit_status.json').read_text())
    run = json.loads((OUTPUT / 'run.json').read_text())
    ledger = json.loads((OUTPUT / 'gradient_ledger.json').read_text())
    rows = [json.loads(line) for line in (OUTPUT / 'gradient_horizons.jsonl').read_text().splitlines()]
    assert status['state'] == 'DIAGNOSIS_COMPLETE'
    assert status['control_steps'] == 288 and status['optimizer_updates'] == 0
    assert status['completed_updates'] == 45011
    assert status['state_hashes_before'] == status['state_hashes_after']
    assert status['forbidden_calls'] == []
    assert len(rows) == ledger['horizons'] == 12
    assert not list(OUTPUT.glob('model_*.pt')) and not list(OUTPUT.glob('student_*.pt'))
    checkpoint = Path(run['resume'])
    reference = Path(run['startup_anchor']['reference'])
    assert sha256(checkpoint) == run['resume_sha256']
    assert sha256(reference) == run['startup_anchor']['reference_sha256']
    gpu_processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert gpu_processes == '', 'A compute process remains; inspect it before finalizing.'
    result = dict(state='FINALIZED_ZERO_UPDATES', completed_updates=45011,
                  optimizer_updates=0, control_steps=288, failed_attempt_control_steps=0,
                  hashes_unchanged=True, input_files_unchanged=True, checkpoint_writes=0,
                  gpu_compute_processes=[], finalized_unix=time.time(),
                  checkpoint_sha256=run['resume_sha256'],
                  reference_sha256=run['startup_anchor']['reference_sha256'],
                  wrapper_success_is_not_training_evidence=True,
                  notes='The generic train wrapper may add one to its SUCCESS counter after a zero-update learn override. Only audit_status and this receipt are authoritative for this diagnostic.')
    with (OUTPUT / 'FINALIZED.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
