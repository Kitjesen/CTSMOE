"""Launch two isolated frozen diagnostics; never edit or stop formal training."""
import json
from pathlib import Path
import subprocess
import sys


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/stair_failure_peak_20260918'
checkpoint = root / 'runs/official4096_30000_straightspeed_20260918/model_19000.pt'
jobs = []
for name, extra in (('guarded', []), ('no_actuator_termination', ['--diagnostic-no-actuator-termination'])):
    output = audit / name
    assert not output.exists()
    argv = [sys.executable, str(audit / 'probe_failure.py'), '--root', str(root), '--checkpoint', str(checkpoint),
            '--output', str(output), '--mode', 'heading', *extra]
    with output.with_suffix('.log').open('x') as stream:
        child = subprocess.Popen(argv, cwd=audit, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    jobs.append({'name': name, 'pid': child.pid, 'argv': argv})
(audit / 'launch.json').write_text(json.dumps(jobs, indent=2))
print(json.dumps(jobs))
