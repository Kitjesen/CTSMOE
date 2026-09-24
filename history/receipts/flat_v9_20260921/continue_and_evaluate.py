"""Complete one thousand V9 repair updates, then run fixed multi-terrain evaluation."""
import json
from pathlib import Path
import subprocess
import sys

root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
code=root/'code_flat_v9_20260921'
smoke=root/'runs/flat_v9_smoke_20260921'
assert json.loads((smoke/'SUCCESS.json').read_text())['completed_updates']==43727
run=json.loads((smoke/'run.json').read_text())
assert run['resume_checks']['terrain_restored_exactly']
assert not (smoke/'phase_foundation_migration.json').exists()
output=root/'runs/flat_v9_1000_20260921'
assert not output.exists()
subprocess.run([sys.executable,'-u','train.py','--root',str(root),'--output',str(output),
    '--num-envs','4096','--updates','984','--mixed-pit','--startup-repair-profile','omni_start_v9',
    '--resume',str(smoke/'model_43726.pt')],cwd=code,check=True)
assert json.loads((output/'SUCCESS.json').read_text())['completed_updates']==44711
selection=[dict(path='runs/startup_repair_flat40_96_20260920/model_43201.pt',label='V6_43202',teacher=False),
    dict(path='runs/phase_v8_1000_20260921/model_43710.pt',label='V8_43711',teacher=False),
    dict(path='runs/flat_v9_1000_20260921/model_44710.pt',label='V9_44711',teacher=False),
    dict(path='runs/flat_v9_1000_20260921/model_44710.pt',label='V9_44711_teacher',teacher=True)]
selection_path=root/'flat_v9_selection_20260921.json'
with selection_path.open('x') as stream:
    json.dump(selection,stream,indent=2)
for script,destination in [('all_scene_phase_audit_v2.py','flat_v9_original_20260921'),
                           ('flat_v9_stage_20260921/evaluate_v9.py','flat_v9_landing_20260921')]:
    subprocess.run([sys.executable,'-u',str(root/script),'--root',str(root),'--code',str(code),
        '--selection',str(selection_path),'--output',str(root/'recording'/destination)],cwd=code,check=True)
print('FLAT_V9_TRAINING_AND_EVALUATIONS_COMPLETE',flush=True)
