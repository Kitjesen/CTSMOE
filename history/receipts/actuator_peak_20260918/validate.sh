#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
PY=/root/autodl-tmp/conda/envs/isaaclab/bin/python
A="$R/recording/actuator_peak_20260918"
cd "$R/code_peak_20260918"
export PYTHONUNBUFFERED=1
"$PY" train.py --root "$R" --output "$A/smoke64" --num-envs 64 --updates 4 \
  --resume "$R/recording/actuator_guard_20260918/export_resume64/model_83.pt" \
  --actuator-profile rs0402_peak_v2 > "$A/smoke64.log" 2>&1
"$PY" train.py --root "$R" --output "$A/resume64" --num-envs 64 --updates 2 \
  --resume "$A/smoke64/model_87.pt" > "$A/resume64.log" 2>&1
"$PY" record_student.py --root "$R" --policy "$A/resume64/student_90.pt" \
  --output "$A/record_smoke" --smoke --trace-only > "$A/record_smoke.log" 2>&1
"$PY" verify_peak.py
"$PY" probe_peak.py --root "$R" --checkpoint "$R/runs/official4096_30000_straightspeed_20260918/model_19000.pt" \
  --output "$A/heading19001_peak" --mode heading > "$A/heading19001_peak.log" 2>&1
echo PEAK_VALIDATION_AND_FROZEN_PROBE_COMPLETE
