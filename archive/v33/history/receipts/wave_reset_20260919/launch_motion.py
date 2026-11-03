"""A frozen-policy placement test; compare against the prior matched rate12 arm."""
import json
from pathlib import Path
import subprocess
import sys

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root/'recording/wave_reset_20260919'
args = [sys.executable, str(folder/'probe_motion.py'), '--root', str(root),
        '--checkpoint', str(root/'runs/official4096_30000_rs0402peak_20260918/model_25500.pt'),
        '--output', str(folder/'surface_rate12'), '--target-rate', '12']
with (folder/'surface_rate12.log').open('x') as log:
    process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
(folder/'motion_launch.json').write_text(json.dumps(dict(pid=process.pid, command=args), indent=2))
print(process.pid)
