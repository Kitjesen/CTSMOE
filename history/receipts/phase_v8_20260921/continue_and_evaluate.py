"""Resume verified V8 smoke to 1000 total updates, then run both terrain suites."""
import json
from pathlib import Path
import subprocess
import sys

root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
code=root/'code_phase_v8_20260921'
smoke=root/'runs/phase_v8_smoke_20260921'
assert json.loads((smoke/'SUCCESS.json').read_text())['completed_updates']==43218
output=root/'runs/phase_v8_1000_20260921'
assert not output.exists()
subprocess.run([sys.executable,'-u','train.py','--root',str(root),'--output',str(output),
    '--num-envs','4096','--updates','984','--mixed-pit','--startup-repair-profile','omni_start_v8',
    '--resume',str(smoke/'model_43217.pt')],cwd=code,check=True)
assert json.loads((output/'SUCCESS.json').read_text())['completed_updates']==44202
selection=[dict(path='runs/startup_repair_flat40_96_20260920/model_43201.pt',label='V6_43202',teacher=False),
           dict(path='runs/phase_v8_1000_20260921/model_44201.pt',label='V8_44202',teacher=False),
           dict(path='runs/phase_v8_1000_20260921/model_44201.pt',label='V8_44202_teacher',teacher=True)]
selection_path=root/'phase_v8_selection_20260921.json'
with selection_path.open('x') as stream:
    json.dump(selection,stream,indent=2)
for script, destination in [('all_scene_phase_audit_v2.py','phase_v8_original_20260921'),
                            ('phase_v8_stage_20260921/evaluate_landing.py','phase_v8_landing_20260921')]:
    subprocess.run([sys.executable,'-u',str(root/script),'--root',str(root),'--code',str(code),
                    '--selection',str(selection_path),'--output',str(root/'recording'/destination)],check=True,cwd=code)
print('PHASE_V8_TRAINING_AND_BOTH_EVALUATIONS_COMPLETE',flush=True)
