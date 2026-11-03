"""Install new V22 driver/assessment only; never overwrite frozen training source."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922')
HERE = ROOT/'hip_guard_v22_attempt02'
payload = Path('/tmp/hip_guard_v22_orchestration.zip')
with zipfile.ZipFile(payload) as z:
    for item in z.infolist():
        path = HERE/item.filename
        if Path(item.filename).is_absolute() or '..' in Path(item.filename).parts or path.exists():
            raise ValueError(f'Unsafe or existing output: {item.filename}')
    z.extractall(HERE)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((HERE/'ORCHESTRATION_FILES.json').read_text())
for name,digest in manifest['files'].items():
    if sha(HERE/name) != digest:
        raise ValueError(f'Upload changed: {name}')
dependencies = [*Path('/root/autodl-tmp/cts-v15-evaluation-audit-20260922').glob('*.py'),
                * (ROOT/'speed_brake_v20/landing_audit_v21').glob('*.py'),
                ROOT/'stair_coverage_v21/evaluation_34823/terrain83/manifest.json']
manifest['external'] = {str(p):sha(p) for p in dependencies}
with (HERE/'ORCHESTRATION_V22.json').open('x') as f:
    json.dump(manifest,f,indent=2)
print('ORCHESTRATION_INSTALLED',len(manifest['files']),len(manifest['external']))
