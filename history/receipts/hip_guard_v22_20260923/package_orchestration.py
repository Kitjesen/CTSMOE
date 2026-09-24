import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ASSESSMENT = HERE/'assessment'
for name in ('run_assessment.py', 'hip_metrics.py'):
    shutil.copy2(HERE/name,ASSESSMENT/name)
files = {'driver_v22.py': HERE/'driver_v22.py'}
files.update({f'assessment/{p.name}':p for p in ASSESSMENT.glob('*') if p.is_file()})
manifest = {name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
(HERE/'ORCHESTRATION_FILES.json').write_text(json.dumps(dict(files=manifest),indent=2),encoding='utf-8')
files['ORCHESTRATION_FILES.json'] = HERE/'ORCHESTRATION_FILES.json'
with zipfile.ZipFile(HERE/'orchestration_payload.zip','x',zipfile.ZIP_DEFLATED) as z:
    for name,p in files.items():
        z.write(p,name)
print(hashlib.sha256((HERE/'orchestration_payload.zip').read_bytes()).hexdigest())
