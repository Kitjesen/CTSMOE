"""Run bounded validation alongside, not inside, the existing trainer."""
from pathlib import Path
import subprocess
import sys

folder = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/wave_reset_20260919')
with (folder/'geometry.log').open('x') as log:
    process = subprocess.Popen([sys.executable, str(folder/'validate_geometry.py')], stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
print(process.pid)
