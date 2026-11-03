#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
export PYTHONUNBUFFERED=1
for mode in training controlled; do
  "$P" "$R/slope_audit_probe_v2.py" --root "$R" --code "$R/code_flat_speed_20260920_v1" --checkpoint "$R/runs/mixed9_4096_60000_individualpit_20260919/model_37282.pt" --output "$R/runs/slope_audit_${mode}_20260920_v2" --mode "$mode"
done
