"""Read-only paired evaluation after a user-requested saved training stop."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
AUDIT = Path('/root/autodl-tmp/cts-v15-evaluation-audit-20260922')
V15 = Path('/root/cts-early-ablation-v15-20260921')
SOURCE = BASE / 'runs/mixed9_4096_60000_relax45_20260919/model_31830.pt'
CANDIDATE = Path('/root/autodl-tmp/cts-uphill-runup-v16-20260922/runs/completed_33831/model_33033.pt')


def main():
    with (BASE / 'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if gpu.strip():
            raise RuntimeError('GPU is occupied; no parallel evaluation')
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE, CANDIDATE)}
        protocol = json.loads((AUDIT / 'protocol.json').read_text())
        protocol['sources'] = [dict(path=str(SOURCE), completed_updates=31831),
                               dict(path=str(CANDIDATE), completed_updates=33034)]
        selection = [dict(path=str(p), label=label + ('_teacher' if teacher else ''), teacher=teacher)
                     for p, label in ((SOURCE, 'early_31831'), (CANDIDATE, 'saved_33034'))
                     for teacher in (False, True)]
        for name, value in [('protocol.json', protocol), ('selection.json', selection), ('hashes_before.json', hashes)]:
            with (HERE / name).open('x') as f:
                json.dump(value, f, indent=2)
        output = HERE / 'flat25_33034'
        subprocess.run([sys.executable, '-u', str(AUDIT / 'evaluate_corrected.py'),
                        '--root', str(BASE), '--code', str(V15 / 'code'), '--output', str(output),
                        '--selection', str(HERE / 'selection.json'), '--protocol', str(HERE / 'protocol.json'),
                        '--suite', 'flat25'], cwd=V15 / 'code', check=True)
        subprocess.run([sys.executable, str(V15 / 'analyze.py'), str(output)], check=True)
        manifest = json.loads((output / 'manifest.json').read_text())
        assert len(manifest['cases']) == 80 and manifest['optimizer_steps'] == 0
        assert manifest['completed_updates'] == 33034
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
        (HERE / 'FINALIZED.json').write_text(json.dumps(dict(state='complete', optimizer_updates=0,
            completed_updates=33034, cases=80, checkpoint_hashes_unchanged=True), indent=2))


if __name__ == '__main__':
    main()
