"""Separate the new hip contract from the strict inherited motor contract."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from build_candidate import replace_once

HERE = Path(__file__).resolve().parent
OUT = HERE/'candidate_attempt03'
shutil.copytree(HERE/'candidate_attempt02',OUT,ignore=shutil.ignore_patterns('__pycache__'))
train = (OUT/'code/train.py').read_text(encoding='utf-8')
if train.count('{**export_contract(actuator_profile), "hip_guard": hip_contract()}') != 2:
    raise ValueError('Unexpected contract serialization locations')
train = train.replace('{**export_contract(actuator_profile), "hip_guard": hip_contract()}',
                      'export_contract(actuator_profile)')
train = replace_once(train, '                        "actuator_contract.json": json.dumps(export_contract(actuator_profile)),',
    '                        "actuator_contract.json": json.dumps(export_contract(actuator_profile)),\n'
    '                        "hip_contract.json": json.dumps(hip_contract()),')
train = replace_once(train, '        configure_hip(cfg)\n',
    '        if saved.get("hip_guard_profile") == HIP_PROFILE and saved.get("hip_contract") != hip_contract():\n'
    '            raise ValueError("Saved hip configuration differs from this continuation")\n'
    '        configure_hip(cfg)\n')
(OUT/'code/train.py').write_text(train,encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


p=json.loads((OUT/'PREPARED.json').read_text())
p['files']['train.py']=sha(OUT/'code/train.py')
(OUT/'PREPARED.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
m=json.loads((OUT/'RUNTIME_PREPARE_MANIFEST.json').read_text())
m['files']={name:sha(OUT/name) for name in m['files']}
(OUT/'RUNTIME_PREPARE_MANIFEST.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
with zipfile.ZipFile(HERE/'candidate_attempt03_payload.zip','x',zipfile.ZIP_DEFLATED) as z:
    for f in OUT.rglob('*'):
        if f.is_file():
            z.write(f,f.relative_to(OUT).as_posix())
print(sha(HERE/'candidate_attempt03_payload.zip'))
