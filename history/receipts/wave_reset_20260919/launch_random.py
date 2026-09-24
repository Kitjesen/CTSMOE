import json
from pathlib import Path
import subprocess
import sys

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root/'recording/wave_reset_20260919'
for name, profile in [('random_original', 'original'), ('random_surface', 'wave_surface_v1')]:
    args = [sys.executable, str(folder/'probe_random_start.py'), '--root', str(root),
            '--checkpoint', str(root/'runs/official4096_30000_rs0402peak_20260918/model_25500.pt'),
            '--output', str(folder/name), '--target-rate', '12', '--wave-reset-profile', profile]
    with (folder/f'{name}.log').open('x') as log:
        child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (folder/f'{name}_launch.json').write_text(json.dumps(dict(pid=child.pid, command=args), indent=2))
    print(name, child.pid)
