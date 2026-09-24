"""Install the isolated default-only correction without overwriting prior runs."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922')
HERE = ROOT/'hip_default_v23_20260923'
payload = Path('/tmp/hip_default_v23_20260923.zip')
expected = '8029e1fe05c38a1d969cfa484a33436b0915672c326fd378f147958a5ea2987d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha(payload) != expected or HERE.exists():
    raise ValueError('Wrong payload or destination already exists')
stopped = json.loads((ROOT/'hip_guard_v22_attempt03/pipeline_status.json').read_text())
if stopped['completed_updates'] != 34925 or stopped.get('training_running') is not False:
    raise ValueError('V22 stop/budget was not verified')
with zipfile.ZipFile(payload) as archive:
    for item in archive.infolist():
        path = Path(item.filename)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe archive member')
    archive.extractall(HERE)
manifest = json.loads((HERE/'ORCHESTRATION_FILES.json').read_text())
for name,digest in manifest['files'].items():
    if sha(HERE/name) != digest:
        raise ValueError(f'Upload changed: {name}')
dependencies = [*Path('/root/autodl-tmp/cts-v15-evaluation-audit-20260922').glob('*.py'),
                *(ROOT/'speed_brake_v20/landing_audit_v21').glob('*.py'),
                ROOT/'stair_coverage_v21/evaluation_34823/terrain83/manifest.json']
manifest['external'] = {str(p):sha(p) for p in dependencies}
(HERE/'ORCHESTRATION_V23.json').write_text(json.dumps(manifest,indent=2))
sys.path.insert(0, str(HERE))
from runtime_contract import verify_prepared  # noqa: E402
verify_prepared(HERE)
print('INSTALLED_DEFAULT_ONLY_V23_NOT_STARTED')
