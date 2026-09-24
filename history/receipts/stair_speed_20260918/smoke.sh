#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
PY=/root/autodl-tmp/conda/envs/isaaclab/bin/python
A="$R/recording/stair_speed_20260918"
cd "$R/code"
export PYTHONUNBUFFERED=1
"$PY" train_straight_speed_20260918.py --root "$R" --output "$A/smoke64" \
  --num-envs 64 --updates 60 --resume "$R/recording/stair_focus_20260918/resume64/model_7.pt" \
  --stair-command-profile straight_speed_v1 > "$A/smoke64.log" 2>&1
"$PY" train_straight_speed_20260918.py --root "$R" --output "$A/resume64" \
  --num-envs 64 --updates 4 --resume "$A/smoke64/model_67.pt" > "$A/resume64.log" 2>&1
"$PY" "$A/verify_smoke.py"
echo STRAIGHT_STAIR_SMOKE_AND_RESUME_COMPLETE
