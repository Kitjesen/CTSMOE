"""Serial common nine-terrain, all-direction and platform/15cm stair assessment."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
ROOT = HERE.parent.parent
SOURCE_PROTOCOL = ROOT/'stair_coverage_v21/evaluation_34823/terrain83/manifest.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with (BASE/'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied')
        args.output.mkdir(exist_ok=False)
        protocol = json.loads(SOURCE_PROTOCOL.read_text())['protocol_config']
        protocol['sources'] = []
        selection, hashes = [], {}
        for item in args.sources:
            label, completed, path = item.split(':', 2)
            protocol['sources'].append(dict(path=path, completed_updates=int(completed)))
            hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for teacher in (False, True):
                selection.append(dict(path=path, label=label+('_teacher' if teacher else ''), teacher=teacher))
        for name, data in [('protocol.json',protocol), ('selection.json',selection), ('HASHES_BEFORE.json',hashes)]:
            (args.output/name).write_text(json.dumps(data, indent=2))
        for suite, stairs15 in [('terrain83',False), ('flat25',False), ('stairend',False), ('stairend',True)]:
            label = suite+('_15cm' if stairs15 else '')
            evaluator = HERE/('evaluate_hip_platform.py' if suite == 'stairend' else 'evaluate_hip.py')
            cmd = [sys.executable,'-u',str(evaluator),'--root',str(BASE),'--code',str(HERE.parent/'code'),
                   '--output',str(args.output/label),'--protocol',str(args.output/'protocol.json'),
                   '--selection',str(args.output/'selection.json'),'--suite',suite]
            if stairs15:
                cmd.append('--stairs15')
            if args.smoke:
                cmd.append('--smoke')
            with (args.output/f'{label}.log').open('x') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable,str(HERE/'hip_metrics.py'),str(args.output/label)], check=True)
        if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items()):
            raise RuntimeError('Source checkpoint mutated during evaluation')
        (args.output/'FINALIZED.json').write_text(json.dumps(dict(state='complete',optimizer_updates=0,
            hashes_unchanged=True, smoke_only=args.smoke, policy_count=len(selection))))


if __name__ == '__main__':
    main()
