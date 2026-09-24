"""Freeze model_23500 and test straight commands with/without heading feedback."""
import json
from pathlib import Path
import subprocess
import sys


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/motion_quality_20260918'
checkpoint = root / 'runs/official4096_30000_rs0402peak_20260918/model_23500.pt'
jobs = []
for mode in ('baseline', 'heading'):
    output = audit / mode
    assert not output.exists()
    argv = [sys.executable, str(audit / 'probe_motion.py'), '--root', str(root),
            '--checkpoint', str(checkpoint), '--output', str(output), '--mode', mode]
    with output.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=audit, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    jobs.append({'mode': mode, 'pid': child.pid, 'argv': argv})
(audit / 'launch.json').write_text(json.dumps(jobs, indent=2))
print(json.dumps(jobs))
