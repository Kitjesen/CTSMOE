"""Wait for the already running zero-update qualification, verify, then run V21 once."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def status(state, **fields):
    tmp = HERE/f'qualification_transition.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(dict(state=state, transition_pid=os.getpid(),
        utc=datetime.now(timezone.utc).isoformat(), **fields), indent=2))
    tmp.replace(HERE/'qualification_transition.json')


def main():
    qualification_pid = int(sys.argv[1])
    with (HERE/'TRANSITION_STARTED.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), qualification_pid=qualification_pid), stream)
    deadline = time.monotonic()+2400
    status('waiting_existing_qualification', qualification_pid=qualification_pid)
    while not (HERE/'READY_FOR_REVIEW.json').exists():
        if not Path(f'/proc/{qualification_pid}').exists():
            raise RuntimeError('Qualification exited without complete zero-update receipt')
        if time.monotonic() > deadline:
            raise TimeoutError('Qualification wait exceeded40min; no training started')
        time.sleep(5)
    status('independent_verification')
    with (HERE/'independent_verification.log').open('x') as log:
        subprocess.run([sys.executable, str(HERE/'verify_preflight.py')],
                       cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)
    # The driver independently checks source, preflight, showcase, files and locks.
    with (HERE/'driver.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(HERE/'driver.py')],
                                 cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        status('training_and_evaluation', supervisor_pid=child.pid)
        rc = child.wait()
    if rc:
        raise RuntimeError(f'V21 driver exited{rc}; preserve state; no retry')
    status('bounded_run_stopped', pipeline=json.loads((HERE/'pipeline_status.json').read_text()))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        status('stopped_on_error', error=str(exc), automatic_retry=False)
        raise
