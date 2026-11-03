"""Create a pinned upload manifest; do not mutate the server."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
REMOTE = '/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917'
old = json.loads((HERE.parent/'full_episode_v12_20260921/source_hashes.json').read_text())
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob('*.py')}
allowed = {'train.py', 'startup_repair.py', 'startup_sampling.py', 'reset_targets.py',
           'test_startup_repair.py', 'test_stair_speed_logic.py', 'test_stance_strength.py'}
changed = {name: value for name, value in hashes.items() if old.get(name) != value}
assert set(changed) <= allowed, changed
assert set(old) <= set(hashes)
with (HERE/'expected_changes.json').open('x') as stream:
    json.dump(changed, stream, indent=2)
with (HERE/'source_hashes.json').open('x') as stream:
    json.dump(hashes, stream, indent=2)
uploads = [[str(CODE/name), f'{REMOTE}/code_startup_v13_20260921/{name}'] for name in hashes]
uploads += [[str(p), f'{REMOTE}/code_startup_v13_20260921/tests/{p.name}'] for p in (CODE/'tests').glob('*.py')]
(HERE/'source_upload.json').write_text(json.dumps(dict(
    mkdir=[f'{REMOTE}/code_startup_v13_20260921/tests', f'{REMOTE}/startup_v13_stage_20260921/preflight'],
    put=uploads), indent=2))
print(json.dumps({'source_files':len(hashes), 'changed':list(changed), 'uploads':len(uploads)}))
