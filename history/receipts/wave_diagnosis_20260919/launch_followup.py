"""Queue isolated contact-start and termination diagnostics, keeping v2 fixed."""

import json
from pathlib import Path
import subprocess
import sys


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root / 'recording/wave_diagnosis_20260919'
checkpoint = root / 'runs/official4096_30000_rs0402peak_20260918/model_25500.pt'
for name, extra in [('lower_reset', ['--reset-z', '-0.1']),
                    ('no_stop', ['--diagnostic-no-actuator-termination'])]:
    args = [sys.executable, str(folder / 'probe_wave.py'), '--root', str(root),
            '--checkpoint', str(checkpoint), '--output', str(folder / name)] + extra
    (folder / f'{name}_command.json').write_text(json.dumps(args, indent=2))
    with (folder / f'{name}.log').open('x') as log:
        subprocess.run(args, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)
print('FOLLOWUP_COMPLETE', flush=True)
