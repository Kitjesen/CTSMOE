"""Zero-update verification of the repaired evaluation wave-reset event."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
BASE = Path('/root/cts-v13-evaluation-recovery-20260921')
STAGE = ROOT / 'startup_v13_stage_20260921'
HERE = BASE / 'wave_reset_parity_r3'
OUTPUT = HERE / 'output'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (ROOT / 'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError('GPU occupied')
        if OUTPUT.exists() or (HERE / 'receipt.json').exists():
            raise FileExistsError('No repeated run or overwrite')
        if shutil.disk_usage(HERE).free < 3 * 1024**3:
            raise RuntimeError('Insufficient space')
        selection = BASE / 'selection_normal.json'
        checkpoints = {str(ROOT / item['path']) for item in json.loads(selection.read_text())}
        before = {path: sha(Path(path)) for path in checkpoints}
        receipt = dict(state='running', started_unix=time.time(), optimizer_updates=0,
                       single_change='preserve deterministic wave_surface reset event',
                       checkpoint_hashes=before)
        (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        subprocess.run([sys.executable, '-u', str(HERE / 'evaluate_wave_parity_r3.py'),
                        '--root', str(ROOT), '--code', str(ROOT / 'code_startup_v13_20260921'),
                        '--selection', str(selection), '--output', str(OUTPUT)], check=True)
        subprocess.run([sys.executable, '-u', str(STAGE / 'analyze.py'), str(OUTPUT)], check=True)
        manifest = json.loads((OUTPUT / 'manifest.json').read_text())
        assert manifest['wave_reset_receipt']['count'] >= 24
        assert 'wave_surface' in manifest['retained_reset_events']
        assert len(manifest['cases']) == 332
        assert all(sha(Path(path)) == value for path, value in before.items())
        receipt.update(state='complete', finished_unix=time.time(),
                       wave_reset_receipt=manifest['wave_reset_receipt'], checkpoints_unchanged=True)
        (HERE / 'FINALIZED.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
