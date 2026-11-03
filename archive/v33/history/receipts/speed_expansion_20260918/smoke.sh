#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
PY=/root/autodl-tmp/conda/envs/isaaclab/bin/python
AUDIT="$R/recording/speed_expansion_20260918"
cd "$R/code"
export PYTHONUNBUFFERED=1
"$PY" train_speed_expanded_20260918.py --root "$R" --output "$AUDIT/smoke64" \
  --num-envs 64 --updates 2 --command-curriculum-start intermediate > "$AUDIT/smoke64.log" 2>&1
"$PY" train_speed_expanded_20260918.py --root "$R" --output "$AUDIT/resume64" \
  --num-envs 64 --updates 2 --resume "$AUDIT/smoke64/model_1.pt" > "$AUDIT/resume64.log" 2>&1
echo SMOKE_AND_RESUME_COMPLETE
