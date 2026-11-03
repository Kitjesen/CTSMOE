from pathlib import Path
import subprocess
import sys

folder = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/wave_reset_20260919')
with (folder/'training_validation.log').open('x') as log:
    child = subprocess.Popen([sys.executable, str(folder/'validate_training.py')], stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
print(child.pid)
