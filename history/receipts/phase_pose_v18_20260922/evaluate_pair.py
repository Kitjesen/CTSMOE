"""Serialized zero-update paired terrain and full-episode flat assessment."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
AUDIT = Path('/root/autodl-tmp/cts-v15-evaluation-audit-20260922')
ANALYSIS = Path('/root/cts-early-ablation-v15-20260921')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', nargs='+', required=True, help='label:completed:absolute_path')
    args = parser.parse_args()
    out = args.output
    with (BASE / 'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid', '--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU occupied')
        out.mkdir(parents=True, exist_ok=False)
        protocol = json.loads((AUDIT / 'protocol.json').read_text())
        protocol['sources'], selection, hashes = [], [], {}
        for encoded in args.sources:
            label, count, path = encoded.split(':', 2)
            protocol['sources'].append(dict(path=path, completed_updates=int(count)))
            for teacher in (False, True):
                selection.append(dict(path=path, label=label+('_teacher' if teacher else ''), teacher=teacher))
            hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, value in [('protocol.json', protocol), ('selection.json', selection), ('hashes_before.json', hashes)]:
            (out / name).write_text(json.dumps(value, indent=2))
        for suite in ('terrain83','flat25'):
            steps = [([sys.executable,'-u',str(AUDIT/'evaluate_corrected.py'),
                '--root',str(BASE),'--code','/root/autodl-tmp/cts-posture-stop-v17-20260922/code',
                '--output',str(out/suite),'--protocol',str(out/'protocol.json'),
                '--selection',str(out/'selection.json'),'--suite',suite], 'evaluate'),
                ([sys.executable,str(ANALYSIS/'analyze.py'),str(out/suite)], 'analyze'),
                ([sys.executable,str(ANALYSIS/'enrich_analysis.py'),str(out/suite)], 'enrich')]
            for command, label in steps:
                (out/'status.json').write_text(json.dumps(dict(state='running', stage=f'{suite}_{label}', optimizer_updates=0)))
                with (out/f'{suite}_{label}.log').open('x') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path,expected in hashes.items())
        (out/'FINALIZED.json').write_text(json.dumps(dict(state='complete',optimizer_updates=0,
            hashes_unchanged=True, terrain_cases=83*len(selection),flat_cases=20*len(selection))))
        (out/'status.json').write_text(json.dumps(dict(state='complete', optimizer_updates=0)))


if __name__ == '__main__':
    main()
