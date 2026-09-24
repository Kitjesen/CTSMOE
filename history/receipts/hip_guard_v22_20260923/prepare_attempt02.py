import hashlib
import json
from pathlib import Path
import shutil
import zipfile

root = Path(__file__).resolve().parent
candidate = root/'candidate_attempt02'
shutil.copytree(root/'candidate', candidate, ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(root/'hip_guard.py', candidate/'code/hip_guard.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


prepared = json.loads((candidate/'PREPARED.json').read_text())
prepared['files']['hip_guard.py'] = sha(candidate/'code/hip_guard.py')
(candidate/'PREPARED.json').write_text(json.dumps(prepared, indent=2), encoding='utf-8')
manifest = json.loads((candidate/'RUNTIME_PREPARE_MANIFEST.json').read_text())
manifest['files'] = {name: sha(candidate/name) for name in manifest['files']}
(candidate/'RUNTIME_PREPARE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
with zipfile.ZipFile(root/'candidate_attempt02_payload.zip', 'x', zipfile.ZIP_DEFLATED) as z:
    for p in candidate.rglob('*'):
        if p.is_file():
            z.write(p, p.relative_to(candidate).as_posix())
print(sha(root/'candidate_attempt02_payload.zip'))
