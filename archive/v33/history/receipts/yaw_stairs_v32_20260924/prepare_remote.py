import hashlib,json,zipfile,subprocess,sys
from pathlib import Path
import torch
root=Path('/root/autodl-tmp/cts-omni-repair-v19-20260922')
stage=root/'yaw_stairs_v32_20260924'
if stage.exists(): raise FileExistsError(stage)
source=root/'omni_landing_v31_20260924/training_38744_to_46194/model_39161.pt'
saved=torch.load(source,map_location='cpu',weights_only=False)
assert saved['completed_updates']==39162
assert 'optimizer_state_dict' in saved and 'optimizer_stu_enc_state_dict' in saved
assert saved['omni_routes_profile']=='omni_routes_v31'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
stage.mkdir()
with zipfile.ZipFile(root/'v32_bundle_20260924.zip') as z: z.extractall(stage)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
here=stage/'orchestration'
receipt=dict(source_path=str(source),source_sha256=digest(source),source_completed_updates=39162,
 code={p.name:digest(p) for p in (stage/'code').glob('*.py')},
 scripts={n:digest(here/n) for n in ('v31_common.py','v31_preflight.py','state_guard.py')})
(here/'V31_FROZEN.json').write_text(json.dumps(receipt,indent=2))
with (here/'coordinator.log').open('x') as log:
 proc=subprocess.Popen([sys.executable,'-u',str(here/'continuous_v31.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(coordinator_pid=proc.pid,stage=str(stage),start=39162,target=46194)))
