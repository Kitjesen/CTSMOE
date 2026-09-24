"""Create an immutable v3 code snapshot, preserving the active v2 directory."""

import hashlib
import json
from pathlib import Path
import shutil


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root / 'recording/wave_diagnosis_20260919'
source = root / 'code_peak_20260918'
target = root / 'code_slew12_20260919'
assert not target.exists()
shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
changed = ['actuator_guard.py', 'policy_contract.py', 'train.py', 'test_actuator_guard.py', 'test_policy_contract.py']
for name in changed:
    shutil.copyfile(folder / 'incoming' / name, target / name)
receipt = {'target': str(target), 'source': str(source), 'changed': changed,
           'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob('*.py')}}
(folder / 'code_snapshot.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
