"""Serialized, zero-update paired flat evaluation after the authorized V17 run."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
AUDIT = Path('/root/autodl-tmp/cts-v15-evaluation-audit-20260922')
SOURCE = BASE / 'runs/mixed9_4096_60000_relax45_20260919/model_31830.pt'
CANDIDATE = HERE / 'training_31831_to_32831/model_32830.pt'
OUT = HERE / 'recording_final_32831'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    print('WAITING_FOR_TRAINING_LOCK', flush=True)
    with (BASE / 'full_episode_v12_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        final = json.loads((HERE / 'training_pipeline_status.json').read_text())
        if final.get('state') != 'training_complete_pending_evaluation' or final.get('completed_updates') != 32831:
            raise RuntimeError('Training did not complete the requested budget; recording not started')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied after training lock release')
        OUT.mkdir(exist_ok=False)
        hashes = {str(p): digest(p) for p in (SOURCE, CANDIDATE)}
        protocol = json.loads((AUDIT / 'protocol.json').read_text())
        protocol['sources'] = [dict(path=str(SOURCE), completed_updates=31831),
                               dict(path=str(CANDIDATE), completed_updates=32831)]
        selection = [dict(path=str(p), label=label + ('_teacher' if teacher else ''), teacher=teacher)
                     for p, label in ((SOURCE, 'source_31831'), (CANDIDATE, 'v17_32831'))
                     for teacher in (False, True)]
        for name, value in [('protocol.json', protocol), ('selection.json', selection),
                            ('hashes_before.json', hashes)]:
            (OUT / name).write_text(json.dumps(value, indent=2))
        args = [sys.executable, '-u', str(AUDIT / 'evaluate_corrected.py'), '--root', str(BASE),
                '--code', str(HERE / 'code'), '--output', str(OUT / 'flat25'),
                '--selection', str(OUT / 'selection.json'), '--protocol', str(OUT / 'protocol.json'),
                '--suite', 'flat25']
        print('RECORDING_PAIRED_80_CASES', flush=True)
        with (OUT / 'evaluation.log').open('x') as log:
            subprocess.run(args, cwd=HERE / 'code', stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((OUT / 'flat25/manifest.json').read_text())
        if len(manifest['cases']) != 80 or manifest['optimizer_steps'] != 0 or manifest['completed_updates'] != 32831:
            raise RuntimeError('Incomplete paired evaluation')
        if any(digest(Path(p)) != h for p, h in hashes.items()):
            raise RuntimeError('Checkpoint changed during evaluation')
        export_forward(manifest)
        (OUT / 'FINALIZED.json').write_text(json.dumps(dict(state='complete', cases=80,
            optimizer_updates=0, source_completed=31831, candidate_completed=32831,
            checkpoint_hashes_unchanged=True), indent=2))
        print('RECORDING_COMPLETE', flush=True)


def export_forward(manifest):
    import numpy as np
    cases = [c.copy() for c in manifest['cases']
             if not c['teacher'] and c['mode'] == 'direct' and c['direction'] == 'forward']
    if len(cases) != 2:
        raise RuntimeError('Expected source and candidate forward episodes')
    ids = [c['id'] for c in cases]
    with np.load(OUT / 'flat25/traces.npz') as data:
        arrays = {k: data[k][:, ids] for k in data.files if k not in ('initial_q', 'initial_root')}
        arrays.update(initial_q=data['initial_q'][ids], initial_root=data['initial_root'][ids])
    for index, case in enumerate(cases):
        case.update(source_case_id=case['id'], id=index)
    manifest.update(cases=cases, source_case_count=80,
                    selection_rule='Both student forward direct first episodes, no outcome selection')
    np.savez_compressed(OUT / 'forward.npz', **arrays)
    (OUT / 'forward_manifest.json').write_text(json.dumps(manifest, indent=2))
    (OUT / 'runtime.urdf').write_bytes((OUT / 'flat25/runtime.urdf').read_bytes())


if __name__ == '__main__':
    main()
