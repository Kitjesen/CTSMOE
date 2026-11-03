"""Extend the not-yet-started evaluation entry point; preserve the old regression."""
import hashlib
import json
import os
from pathlib import Path
import py_compile


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
status = json.loads((root / 'runs/startup_repair_continuous1000_20260920/status.json').read_text())
assert status['state'] == 'training' and status['completed_updates'] < 44106
verification = json.loads((root / 'recording/all_scene_phase_smoke_20260921/verification.json').read_text())
assert verification['cases'] == 213 and verification['smoke_only']
entry = root / 'startup_repair_regression.py'
core = root / 'startup_repair_regression_core_20260921.py'
pending = root / 'regression_with_all_scenes.pending.py'
for path in (entry, pending, root / 'all_scene_phase_audit_v2.py'):
    py_compile.compile(str(path), doraise=True)
original = entry.read_bytes()
assert b'REGRESSION_AND_ALL_SCENES_COMPLETE' not in original
with core.open('xb') as output:
    output.write(original)
assert core.read_bytes() == original
new = pending.read_bytes()
assert b'REGRESSION_AND_ALL_SCENES_COMPLETE' in new
staged = root / 'startup_repair_regression.installing.py'
with staged.open('xb') as output:
    output.write(new)
os.replace(staged, entry)
receipt = dict(completed_updates_at_install=status['completed_updates'],
               scope='Future evaluation only; no change to training or rewards',
               core_sha256=hashlib.sha256(original).hexdigest(),
               dispatcher_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
               smoke_verification=verification,
               observation='Separate simulator runs are not bitwise identical: see measured trace deltas. Telemetry reads are after env.step and do not modify actions or state.')
(root / 'all_scene_dispatch_install_20260921.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
