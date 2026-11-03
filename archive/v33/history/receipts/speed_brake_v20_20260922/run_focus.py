"""Serialized additional zero-update braking and real stair-landing assessment."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
AUDIT = Path('/root/autodl-tmp/cts-v15-evaluation-audit-20260922')
SOURCE = ROOT/'recovery01/training_32831_to_34823/model_33173.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--completed', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with (BASE/'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied')
        args.output.mkdir(parents=True, exist_ok=False)
        protocol = json.loads((AUDIT/'protocol.json').read_text())
        sources = [('source33174', 33174, SOURCE), ('candidate', args.completed, args.candidate)]
        protocol['sources'] = [dict(path=str(p), completed_updates=n) for _, n, p in sources]
        selection = [dict(path=str(p), label=label+('_teacher' if teacher else ''), teacher=teacher)
                     for label, _, p in sources for teacher in (False, True)]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for _, _, p in sources}
        for name, value in [('protocol.json', protocol), ('selection.json', selection), ('hashes_before.json', hashes)]:
            (args.output/name).write_text(json.dumps(value, indent=2))
        for suite in ('brake25', 'stairend'):
            command = [sys.executable, '-u', str(HERE/'evaluate_focus.py'), '--root', str(BASE),
                       '--code', str(HERE/'code'), '--output', str(args.output/suite),
                       '--protocol', str(args.output/'protocol.json'), '--selection', str(args.output/'selection.json'),
                       '--suite', suite]
            if args.smoke:
                command.append('--smoke')
            with (args.output/f'{suite}.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not args.smoke:
                if suite == 'brake25':
                    subprocess.run([sys.executable, str(HERE/'focus_metrics.py'), str(args.output/suite)], check=True)
                else:
                    import numpy as np
                    from focus_protocol import stair_window_metrics
                    folder = args.output/suite
                    manifest = json.loads((folder/'manifest.json').read_text())
                    with np.load(folder/'traces.npz') as arrays:
                        result = stair_window_metrics(manifest, arrays)
                    (folder/'STAIR_END_METRICS.json').write_text(json.dumps(result, indent=2))
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != value for p, value in hashes.items()):
            raise RuntimeError('Evaluation source changed')
        (args.output/'FINALIZED.json').write_text(json.dumps(dict(state='complete', optimizer_updates=0,
            hashes_unchanged=True, smoke_only=args.smoke, candidate_completed=args.completed)))


if __name__ == '__main__':
    main()
