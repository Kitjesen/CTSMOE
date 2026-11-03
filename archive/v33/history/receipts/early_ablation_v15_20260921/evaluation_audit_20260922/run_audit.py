"""Run isolated, zero-update measurements; preserve the original V15 evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
V15 = Path('/root/cts-early-ablation-v15-20260921')
HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def record(state: str, **fields) -> None:
    value = dict(state=state, unix=time.time(), optimizer_updates=0, **fields)
    temporary = HERE / 'status.tmp'
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(HERE / 'status.json')
    print(json.dumps(value), flush=True)


def command(args: list[str], name: str) -> None:
    with (HERE / f'{name}.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen(args, cwd=V15 / 'code', stdout=log,
                                 stderr=subprocess.STDOUT)
        record('running', stage=name, child_pid=child.pid)
        result = child.wait()
    if result != 0:
        raise RuntimeError(f'{name} exited {result}; preserved log and partial output')


def main() -> None:
    if (HERE / 'status.json').exists():
        raise RuntimeError('This audit has already run; use a new attempt directory')
    with (BASE / 'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupied = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True).strip()
        if occupied:
            raise RuntimeError('GPU is occupied; audit will not run alongside another job')
        protocol = json.loads((HERE / 'protocol.json').read_text())
        sources = {row['path']: digest(Path(row['path'])) for row in protocol['sources']}
        frozen_manifest = json.loads((V15 / 'code_hash_manifest.json').read_text())
        (HERE / 'original_code_manifest.json').write_text(json.dumps(frozen_manifest, indent=2))
        (HERE / 'source_hashes.json').write_text(json.dumps(sources, indent=2))
        for suite in ('terrain83', 'flat25'):
            destination = HERE / suite
            command([sys.executable, '-u', str(HERE / 'evaluate_corrected.py'),
                     '--root', str(BASE), '--code', str(V15 / 'code'),
                     '--output', str(destination), '--selection', str(HERE / 'selection.json'),
                     '--protocol', str(HERE / 'protocol.json'), '--suite', suite],
                    f'evaluate_{suite}')
            command([sys.executable, str(V15 / 'analyze.py'), str(destination)], f'analyze_{suite}')
            command([sys.executable, str(V15 / 'enrich_analysis.py'), str(destination)],
                    f'enrich_{suite}')
        for path, expected in sources.items():
            if digest(Path(path)) != expected:
                raise RuntimeError('Evaluation changed a source checkpoint')
        record('complete', stage='paired_evaluation', checkpoint_hashes_unchanged=True)


if __name__ == '__main__':
    if (HERE / 'status.json').exists():
        raise SystemExit('Existing audit status is preserved; use a new attempt directory')
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        record('failed', error=str(error))
        raise
