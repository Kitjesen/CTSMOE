"""Single-GPU bounded migration, qualification and continuation for V33."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from launch_common import LOCK, STAGE, args, digest, frozen
sys.path.insert(0, str(STAGE / 'code'))
from migrate_routes import migrate  # noqa: E402

TARGET = 46194
OLDOUT = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922/env8192_v32_20260924/attempt02/continuation')


def status(state, **values):
    path = STAGE / 'STATUS.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state, **values), indent=2))
    tmp.replace(path)


def run(script, argv, phase):
    frozen()
    with (STAGE / f'{phase}.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(script), *argv],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        status(phase, pid=proc.pid, environments=8192, target=TARGET)
        if proc.wait():
            raise RuntimeError(f'{phase} failed; preserved log and source')
    frozen()


def train(source, output, updates, phase):
    start = torch.load(source, map_location='cpu', weights_only=False)['completed_updates']
    if updates <= 0 or start + updates > TARGET:
        raise ValueError('Invalid remaining update budget')
    run(STAGE / 'code/train.py', args(source, output, updates), phase)
    result = json.loads((output / 'SUCCESS.json').read_text())
    if result['completed_updates'] != start + updates:
        raise RuntimeError('Completed update mismatch')
    checkpoint = output / f'model_{start + updates - 1}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def main():
    if (STAGE / 'STATUS.json').exists():
        raise FileExistsError('Existing attempt cannot be restarted or overwritten')
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if pids.strip():
            raise RuntimeError('GPU occupied; migration requires previous trainer to exit')
        source = max(OLDOUT.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
        saved = torch.load(source, map_location='cpu', weights_only=False)
        start = saved['completed_updates']
        if not 44200 <= start < TARGET - 16:
            raise ValueError('Unexpected migration checkpoint or remaining budget')
        migrated, receipt = migrate(saved)
        destination = STAGE / 'source_migrated.pt'
        torch.save(migrated, destination)
        from state_guard import state_sha256
        if state_sha256(torch.load(destination, map_location='cpu', weights_only=False), torch) != state_sha256(migrated, torch):
            raise RuntimeError('Checkpoint serialization changed migration state')
        receipt.update(source=str(source), source_sha256=digest(source), completed_updates=start,
                       destination=str(destination), target=TARGET, updates_remaining=TARGET-start)
        (STAGE / 'MIGRATION.json').write_text(json.dumps(receipt, indent=2))
        frozen_receipt = dict(source_path=str(destination), source_sha256=digest(destination),
                              source_completed_updates=start,
                              code={p.name: digest(p) for p in (STAGE / 'code').glob('*.py')})
        (STAGE / 'FROZEN.json').write_text(json.dumps(frozen_receipt, indent=2))
        del saved, migrated
        run(STAGE / 'orchestration/preflight.py', [], 'qualification')
        proof = json.loads((STAGE / 'qualification/FINALIZED.json').read_text())
        if (proof['optimizer_updates'] != 0 or proof['steps'] != 1512
                or proof['before'] != proof['after'] or proof['old_geometry_cells_verified'] != 440
                or proof['source_sha256'] != digest(destination)):
            raise RuntimeError('Independent qualification receipt rejected')
        checkpoint = train(destination, STAGE / 'integration16', 16, 'integration16')
        run(STAGE / 'assessment/assess_v33.py', ['--checkpoint', str(checkpoint),
            '--output', str(STAGE / 'integration_assessment'), '--steps', '3'], 'integration_assessment')
        checkpoint = train(checkpoint, STAGE / 'continuation', TARGET-start-16, 'training')
        run(STAGE / 'assessment/assess_v33.py', ['--checkpoint', str(checkpoint),
            '--output', str(STAGE / 'final_assessment')], 'final_assessment')
        status('complete_pending_review', completed=TARGET, checkpoint=str(checkpoint), hardware_release=False)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status('technical_failure', error=traceback.format_exc())
        raise
