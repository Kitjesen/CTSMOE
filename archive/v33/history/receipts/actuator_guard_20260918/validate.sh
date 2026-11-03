#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
PY=/root/autodl-tmp/conda/envs/isaaclab/bin/python
A="$R/recording/actuator_guard_20260918"
C="$R/code_guard_20260918"
export PYTHONUNBUFFERED=1
cd "$C"
"$PY" train.py --root "$R" --output "$A/smoke64" --num-envs 64 --updates 8 \
  --resume "$R/recording/stair_speed_20260918/resume64/model_71.pt" \
  --actuator-profile rs0402_guard_v1 > "$A/smoke64.log" 2>&1
"$PY" train.py --root "$R" --output "$A/resume64" --num-envs 64 --updates 2 \
  --resume "$A/smoke64/model_79.pt" > "$A/resume64.log" 2>&1
"$PY" probe_guard.py --root "$R" --checkpoint "$R/runs/official4096_30000_straightspeed_20260918/model_19000.pt" \
  --output "$A/heading19001_guard" --mode heading > "$A/heading19001_guard.log" 2>&1
echo GUARD_VALIDATION_COMPLETE
