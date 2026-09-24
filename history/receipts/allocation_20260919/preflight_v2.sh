#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R/code_allocation_20260919_v2"
export PYTHONUNBUFFERED=1
"$P" train.py --root "$R" --output "$R/runs/allocation_smoke_v2_20260919" --num-envs 4096 --updates 16 --mixed-pit --allocation-profile balanced_adaptive_v1 --resume "$R/runs/mixed9_4096_60000_climbfirst_20260919/model_33348.pt"
"$P" train.py --root "$R" --output "$R/runs/allocation_inherit_v2_20260919" --num-envs 4096 --updates 2 --mixed-pit --resume "$R/runs/allocation_smoke_v2_20260919/model_33364.pt"
"$P" reset_boundary_probe.py --root "$R" --output "$R/runs/allocation_reset_probe_v2_20260919" --num-envs 4096 --updates 16 --mixed-pit --resume "$R/runs/allocation_inherit_v2_20260919/model_33366.pt"
