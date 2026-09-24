"""Finite V18 pipeline: qualify, train 1000, assess, and stop without continuation."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')


def status(state, **values):
    temporary = HERE / 'pipeline_status.tmp'
    temporary.write_text(json.dumps(dict(state=state, supervisor_pid=os.getpid(),
        time_utc=datetime.now(timezone.utc).isoformat(), source_completed=31831,
        maximum_new_updates=1000, target_completed=32831, **values), indent=2))
    temporary.replace(HERE / 'pipeline_status.json')


def run(stage, arguments):
    with (HERE / f'{stage}_driver.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', *arguments], cwd=HERE,
                                 stdout=log, stderr=subprocess.STDOUT)
        status(stage, child_pid=child.pid)
        result = child.wait()
    if result != 0:
        raise RuntimeError(f'{stage} exited {result}; results preserved, no automatic retry')


def main():
    with (HERE / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE / 'pipeline_status.json').exists():
            raise RuntimeError('A pipeline already exists; review saved state before any resume')
        # Baseline uses the same GPU lock. Its completion must precede preflight.
        deadline = time.monotonic() + 1800
        while not (HERE / 'baseline30000_31831/FINALIZED.json').exists():
            status('waiting_baseline')
            if time.monotonic() > deadline:
                raise TimeoutError('Baseline did not complete within 30 minutes')
            time.sleep(10)
        baseline = json.loads((HERE / 'baseline30000_31831/FINALIZED.json').read_text())
        if baseline.get('state') != 'complete' or baseline.get('optimizer_updates') != 0:
            raise RuntimeError('Baseline incomplete')
        # Source choice is based on directly measured forward posture, not iteration count.
        analysis = json.loads((HERE / 'baseline30000_31831/flat25/summary.json').read_text())
        rows = {row['label']: row for row in analysis['forward']}
        if rows['early31831']['first_peak_rms'] >= rows['early30000']['first_peak_rms']:
            raise RuntimeError('New baseline no longer supports the planned source choice; review required')
        run('preflight', [str(HERE / 'run_preflight.py')])
        run('verify_preflight', [str(HERE / 'verify_preflight.py')])
        run('training', [str(HERE / 'launch_training.py')])
        run('evaluation', [str(HERE / 'evaluate_pair.py'), '--output', str(HERE / 'evaluation1000'),
            '--sources',
            'source31831:31831:' + str(BASE / 'runs/mixed9_4096_60000_relax45_20260919/model_31830.pt'),
            'v17_32831:32831:/root/autodl-tmp/cts-posture-stop-v17-20260922/training_31831_to_32831/model_32830.pt',
            'v18_32831:32831:' + str(HERE / 'training_31831_to_32831/model_32830.pt')])
        status('evaluation_complete_pending_review', completed_updates=32831, automatic_continuation=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        # A duplicate launcher must not replace the active owner's state.
        if (HERE / 'pipeline_status.json').exists():
            current = json.loads((HERE / 'pipeline_status.json').read_text())
            if current.get('supervisor_pid') != os.getpid():
                raise
        status('failed', error=str(error), automatic_retry=False)
        raise
