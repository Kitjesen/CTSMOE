"""Run two isolated diagnostics sequentially to limit GPU contention."""
import json
from pathlib import Path
import subprocess

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/drift_cause_20260918'
python = '/root/autodl-tmp/conda/envs/isaaclab/bin/python'
checkpoint = root / 'runs/official4096_30000_straightspeed_20260918/model_19000.pt'
for mode in ('baseline', 'heading'):
    output = audit / f'{mode}19001_v2'
    argv = [python, str(audit / 'probe_v2.py'), '--root', str(root), '--checkpoint', str(checkpoint),
            '--output', str(output), '--mode', mode]
    with output.with_suffix('.log').open('x') as log:
        process = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=False)
    assert process.returncode == 0, (mode, process.returncode)
    assert json.loads((output / 'status.json').read_text())['state'] == 'complete', mode
    (audit / 'pair_status.json').write_text(json.dumps({'completed_mode': mode, 'state': 'running'}))
(audit / 'pair_status.json').write_text(json.dumps({'state': 'complete'}))
