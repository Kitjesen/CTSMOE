"""Stage the wave placement candidate without changing the live training snapshot."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root / 'recording/wave_reset_20260919'
code = root / 'code_wave_reset_20260919'
assert not code.exists()
shutil.copytree(root / 'code_slew12_20260919', code, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
for source in (folder / 'incoming').glob('*.py'):
    shutil.copyfile(source, code / source.name)
(folder / 'code_snapshot.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')}, indent=2))
print(str(code))
