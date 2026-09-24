"""Run baseline and target-rate-only diagnostic with the same frozen checkpoint."""

import json
from pathlib import Path
import subprocess
import sys


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root / 'recording/wave_diagnosis_20260919'
checkpoint = root / 'runs/official4096_30000_rs0402peak_20260918/model_25500.pt'
jobs = []
for name, extra in [('baseline', []), ('rate12', ['--target-rate', '12'])]:
    output = folder / name
    assert not output.exists()
    args = [sys.executable, str(folder / 'probe_wave.py'), '--root', str(root),
            '--checkpoint', str(checkpoint), '--output', str(output)] + extra
    with (folder / f'{name}.log').open('x') as log:
        child = subprocess.Popen(args, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    jobs.append(dict(name=name, pid=child.pid, args=args))
(folder / 'launch_pair.json').write_text(json.dumps(jobs, indent=2))
print(json.dumps(jobs))
