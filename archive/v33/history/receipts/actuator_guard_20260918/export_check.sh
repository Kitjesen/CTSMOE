#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
PY=/root/autodl-tmp/conda/envs/isaaclab/bin/python
A="$R/recording/actuator_guard_20260918"
cd "$R/code_guard_20260918"
export PYTHONUNBUFFERED=1
"$PY" train_v2.py --root "$R" --output "$A/export_resume64" --num-envs 64 --updates 2 \
  --resume "$A/resume64/model_81.pt" > "$A/export_resume64.log" 2>&1
"$PY" record_student.py --root "$R" --policy "$A/export_resume64/student_84.pt" \
  --output "$A/record_guard_smoke" --smoke --trace-only > "$A/record_guard_smoke.log" 2>&1
"$PY" - <<'PY'
import json
from pathlib import Path
from policy_contract import load_student_export, export_contract
A = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/actuator_guard_20260918')
assert json.loads((A/'export_resume64/SUCCESS.json').read_text())['completed_updates'] == 84
policy, contract = load_student_export(A/'export_resume64/student_84.pt')
assert contract == export_contract('rs0402_guard_v1')
manifest = json.loads((A/'record_guard_smoke/manifest.json').read_text())
assert manifest['actuator_contract'] == contract
run = json.loads((A/'export_resume64/run.json').read_text())
assert all(run['resume_checks'].values())
(A/'export_check.json').write_text(json.dumps({'passed': True, 'contract': contract, 'resume_checks': run['resume_checks']}, indent=2))
print('GUARD_EXPORT_AND_RECORDING_CHECK_PASSED')
PY
