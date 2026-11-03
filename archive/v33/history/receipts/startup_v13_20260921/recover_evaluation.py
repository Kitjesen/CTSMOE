"""Re-run only V13 evaluation on the verified free system volume.

The original data volume filled while saving NPZ. Keep that failed artifact and
all checkpoints unchanged. Exclusive output paths and the existing pipeline lock
prevent overwrites and concurrent GPU work. This script has no training calls.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
STAGE = ROOT / 'startup_v13_stage_20260921'
OUTPUT = Path('/root/cts-v13-evaluation-recovery-20260921')
sys.path.insert(0, str(STAGE))
import run_pipeline as pipeline  # noqa: E402
from gates import continuation_gate, flat25_gate  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(state, **details):
    data = dict(state=state, updated_unix=time.time(), completed_updates=45011,
                training_running=False, optimizer_updates=0,
                automatic_further_training=False, **details)
    temp = OUTPUT / 'pipeline_status.tmp'
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(OUTPUT / 'pipeline_status.json')
    print(json.dumps(data), flush=True)


def main():
    with (ROOT / 'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'
        ], text=True).strip()
        if busy:
            raise RuntimeError(f'GPU occupied: {busy}')
        if shutil.disk_usage(OUTPUT).free < 5 * 1024**3:
            raise RuntimeError('Need at least 5 GiB free on evaluation volume')
        pipeline.verify_preconditions()
        checkpoint = ROOT / 'runs/startup_v13_pilot300_20260921/model_45010.pt'
        success = json.loads((checkpoint.parent / 'SUCCESS.json').read_text())
        if success['completed_updates'] != 45011:
            raise RuntimeError('Candidate training is not complete')
        hashes = {str(p): digest(p) for p in (checkpoint, pipeline.SOURCE, pipeline.REFERENCE)}
        receipt = dict(reason='ENOSPC while saving original evaluation traces',
                       preserved_failed_output=str(ROOT / 'recording/startup_v13_pilot300_normal_20260921'),
                       checkpoints=hashes, scripts={n: digest(STAGE / n) for n in
                       ('evaluate.py', 'analyze.py', 'joint_assessment.py', 'gates.py', 'plot_comparison.py')})
        pipeline.write_json_new(OUTPUT / 'recovery_receipt.json', receipt)
        reports = []
        for flat25 in (False, True):
            name = 'flat25' if flat25 else 'normal'
            selection = OUTPUT / f'selection_{name}.json'
            pipeline.write_json_new(selection, pipeline.selection(checkpoint, 45011, flat25))
            destination = OUTPUT / name
            if destination.exists():
                raise FileExistsError(destination)
            status('evaluating_' + name, output=str(destination))
            args = ['--root', ROOT, '--code', pipeline.CODE, '--selection', selection,
                    '--output', destination]
            if flat25:
                args.append('--flat25')
            pipeline.run('evaluate.py', *args)
            pipeline.run('analyze.py', destination)
            pipeline.run('plot_comparison.py', destination)
            manifest = json.loads((destination / 'manifest.json').read_text())
            report = json.loads((destination / 'analysis.json').read_text())
            expected = 100 if flat25 else 332
            if len(manifest['cases']) != expected or len(report['cases']) != expected:
                raise RuntimeError('Missing evaluation cases')
            reports.append(report)
        for path, before in hashes.items():
            if digest(Path(path)) != before:
                raise RuntimeError('Checkpoint changed during evaluation')
        gates = pipeline.evaluate_gates(*reports, 45011, continuation_gate, flat25_gate)
        pipeline.write_json_new(OUTPUT / 'gate300.json', gates)
        status('pilot300_passed_pending_review' if gates['accepted'] else
               'pilot300_rejected_pending_review', gates=gates,
               normal_evidence=str(OUTPUT / 'normal'), flat25_evidence=str(OUTPUT / 'flat25'))


if __name__ == '__main__':
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        main()
    except BaseException as exc:
        status('evaluation_recovery_failed', error=repr(exc))
        raise
