#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R/code_uphill_speed_20260920_v1"
export PYTHONUNBUFFERED=1
"$P" train.py --root "$R" --output "$R/runs/uphill_speed_smoke_20260920" --num-envs 4096 --updates 4 --mixed-pit --uphill-speed-limit 1.2 --resume "$R/runs/mixed9_4096_60000_flat_speed_20260920/model_37500.pt"
"$P" train.py --root "$R" --output "$R/runs/uphill_speed_inherit_20260920" --num-envs 4096 --updates 2 --mixed-pit --resume "$R/runs/uphill_speed_smoke_20260920/model_37504.pt"
