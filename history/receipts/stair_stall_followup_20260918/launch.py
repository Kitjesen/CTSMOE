"""Evaluate two frozen checkpoints; leave formal training untouched."""
import json
from pathlib import Path
import subprocess
import sys


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/stair_stall_followup_20260918'
jobs = []
for name, run, checkpoint in (
    ('old19001', 'official4096_30000_straightspeed_20260918', 'model_19000.pt'),
    ('current21001', 'official4096_30000_rs0402peak_20260918', 'model_21000.pt'),
):
    output = audit / name
    assert not output.exists()
    argv = [sys.executable, str(audit / 'probe_request.py'), '--root', str(root),
            '--checkpoint', str(root / 'runs' / run / checkpoint),
            '--output', str(output), '--mode', 'heading']
    with output.with_suffix('.log').open('x') as stream:
        child = subprocess.Popen(argv, cwd=audit, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    jobs.append({'name': name, 'pid': child.pid, 'argv': argv})
(audit / 'launch.json').write_text(json.dumps(jobs, indent=2))
print(json.dumps(jobs))
