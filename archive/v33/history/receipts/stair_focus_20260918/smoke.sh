#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
PY=/root/autodl-tmp/conda/envs/isaaclab/bin/python
A="$R/recording/stair_focus_20260918"
cd "$R/code"
export PYTHONUNBUFFERED=1
"$PY" train_stair_focus_20260918.py --root "$R" --output "$A/smoke64" \
  --num-envs 64 --updates 4 --resume "$R/recording/speed_expansion_20260918/smoke64_v2/model_1.pt" \
  --base-height-penalty off --terrain-training-profile stairs80_v1 > "$A/smoke64.log" 2>&1
"$PY" train_stair_focus_20260918.py --root "$R" --output "$A/resume64" \
  --num-envs 64 --updates 2 --resume "$A/smoke64/model_5.pt" > "$A/resume64.log" 2>&1
echo STAIR_FOCUS_SMOKE_AND_RESUME_COMPLETE
