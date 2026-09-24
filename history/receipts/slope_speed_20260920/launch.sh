#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
export PYTHONUNBUFFERED=1
for mode in controlled training; do
  "$P" "$R/slope_speed_probe_v2.py" --root "$R" --code "$R/code_flat_speed_20260920_v1" --checkpoint "$R/runs/mixed9_4096_60000_flat_speed_20260920/model_37500.pt" --output "$R/runs/slope_speed_${mode}_20260920_v2" --mode "$mode"
done
