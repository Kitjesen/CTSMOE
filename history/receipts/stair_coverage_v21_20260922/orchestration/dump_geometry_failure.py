"""Capture the actual rejected terrain; preserve the original exception and no updates."""
from pathlib import Path
import runpy
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'code'))
import stair_transition_coverage as coverage

original = coverage.cap_mesh_arrays


def capture(vertices, faces, kind):
    try:
        return original(vertices, faces, kind)
    except ValueError:
        output = HERE/'geometry_failure_actual.npz'
        if output.exists():
            raise RuntimeError('Failure capture already exists; preserve it') from None
        np.savez_compressed(output, vertices=np.asarray(vertices), faces=np.asarray(faces), kind=kind)
        print('CAPTURED_REJECTED_GEOMETRY', kind, len(vertices), len(faces), flush=True)
        raise


coverage.cap_mesh_arrays = capture
from run_preflight import command
from runtime_contract import verify_prepared

identity, _ = verify_prepared(HERE)
args = command(HERE, identity)
args[args.index('--output')+1] = str(HERE/'geometry_failure_reproduction')
sys.argv = args[2:]
runpy.run_path(str(HERE/'preflight.py'), run_name='__main__')
