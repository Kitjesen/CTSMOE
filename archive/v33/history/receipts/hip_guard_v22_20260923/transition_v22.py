"""Wait for this exact zero-update qualification, then invoke the bounded driver."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

HERE = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922/hip_guard_v22_attempt02')
QUALIFICATION_PID = 76686
with (HERE/'TRANSITION_STARTED.json').open('x') as f:
    json.dump(dict(pid=os.getpid(),qualification_pid=QUALIFICATION_PID),f)
deadline = time.monotonic()+2700
while not (HERE/'READY_FOR_REVIEW.json').exists():
    command = Path(f'/proc/{QUALIFICATION_PID}/cmdline')
    if not command.exists() or str(HERE/'run_preflight.py').encode() not in command.read_bytes():
        raise RuntimeError('Qualification stopped without READY; do not start training')
    if time.monotonic() >= deadline:
        raise TimeoutError('Qualification wait exceeded45min; inspect, do not duplicate')
    (HERE/'transition_status.json').write_text(json.dumps(dict(stage='waiting_zero_update_preflight',
        completed_updates=34823,pid=os.getpid(),time_utc=datetime.now(timezone.utc).isoformat())))
    time.sleep(15)
(HERE/'transition_status.json').write_text(json.dumps(dict(stage='dispatching_independent_verification_and_baseline',
    completed_updates=34823,pid=os.getpid(),time_utc=datetime.now(timezone.utc).isoformat())))
os.execv(sys.executable,[sys.executable,'-u',str(HERE/'driver_v22.py')])
