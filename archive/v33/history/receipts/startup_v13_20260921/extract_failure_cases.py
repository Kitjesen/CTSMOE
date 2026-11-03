"""Copy selected completed evaluation trajectories without simulation or updates."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/cts-v13-evaluation-recovery-20260921')
OUTPUT = ROOT / 'failure_excerpts_r1'
SELECTION = {
    'flat25': [0, 2, 20, 22, 40, 42, 60, 62, 80, 82],
    'normal': [47, 130, 213, 296],
}


def main():
    OUTPUT.mkdir(exist_ok=False)
    for protocol, ids in SELECTION.items():
        path = ROOT / protocol / 'traces.npz'
        manifest = json.loads((ROOT / protocol / 'manifest.json').read_text())
        count = len(manifest['cases'])
        arrays = {}
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                values = data[key]
                if key in ('initial_q', 'initial_root'):
                    if values.shape[0] != count:
                        raise ValueError((key, values.shape))
                    arrays[key] = values[ids]
                else:
                    if values.ndim < 2 or values.shape[1] != count:
                        raise ValueError((key, values.shape))
                    arrays[key] = values[:, ids]
        np.savez_compressed(OUTPUT / f'{protocol}.npz', **arrays)
        manifest['original_case_ids'] = ids
        manifest['cases'] = [manifest['cases'][i] for i in ids]
        manifest['source_trace_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['excerpt_note'] = 'Exact selected case slices; crop by original end_step before analysis.'
        (OUTPUT / f'{protocol}.json').write_text(json.dumps(manifest, indent=2))
        print(protocol, ids, 'saved', flush=True)


if __name__ == '__main__':
    main()
