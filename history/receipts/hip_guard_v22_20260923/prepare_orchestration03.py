import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from build_candidate import replace_once

HERE=Path(__file__).resolve().parent
OUT=HERE/'orchestration03'
OUT.mkdir(exist_ok=False)
shutil.copytree(HERE/'assessment',OUT/'assessment',ignore=shutil.ignore_patterns('__pycache__'))
driver=(HERE/'driver_v22.py').read_text()
driver=replace_once(driver,"if not (HERE/'READY_FOR_REVIEW.json').exists():",
                    "if not (ROOT/'hip_guard_v22_attempt02/READY_FOR_REVIEW.json').exists():")
driver=replace_once(driver,"str(HERE/'verify_preflight.py')","str(HERE/'verify_metadata_bridge.py')")
(OUT/'driver_v22.py').write_text(driver,encoding='utf-8')
shutil.copy2(HERE/'verify_metadata_bridge.py',OUT/'verify_metadata_bridge.py')
files={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
       for p in OUT.rglob('*') if p.is_file()}
(OUT/'ORCHESTRATION_FILES.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')
with zipfile.ZipFile(HERE/'orchestration03_payload.zip','x',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():
            z.write(p,p.relative_to(OUT).as_posix())
installer=(HERE/'install_orchestration.py').read_text()
installer=installer.replace("ROOT/'hip_guard_v22_attempt02'","ROOT/'hip_guard_v22_attempt03'")
installer=installer.replace("'/tmp/hip_guard_v22_orchestration.zip'","'/tmp/hip_guard_v22_orchestration03.zip'")
(HERE/'install_orchestration03.py').write_text(installer,encoding='utf-8')
print(hashlib.sha256((HERE/'orchestration03_payload.zip').read_bytes()).hexdigest())
