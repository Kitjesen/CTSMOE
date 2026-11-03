import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
OUT = HERE / 'candidate'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


prepared = json.loads((OUT/'PREPARED.json').read_text())
prepared['files'] = {p.relative_to(OUT/'code').as_posix(): sha(p) for p in (OUT/'code').rglob('*.py')}
(OUT/'PREPARED.json').write_text(json.dumps(prepared, indent=2), encoding='utf-8')
payload = {p.name: sha(p) for p in OUT.glob('*.py')}
payload.update({n: sha(OUT/n) for n in ('SOURCE_IDENTITY.json', 'PREPARED.json')})
(OUT/'RUNTIME_PREPARE_MANIFEST.json').write_text(json.dumps(dict(files=payload), indent=2), encoding='utf-8')
with zipfile.ZipFile(HERE/'candidate_payload.zip', 'x', zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            z.write(p, p.relative_to(OUT).as_posix())
print('SEALED', sha(HERE/'candidate_payload.zip'))
