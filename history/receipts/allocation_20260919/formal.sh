#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
cd "$R/code_allocation_20260919_v2"
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python train.py --root "$R" --output "$R/runs/mixed9_4096_60000_adaptive_20260919" --num-envs 4096 --updates 26633 --mixed-pit --resume "$R/runs/allocation_inherit_v2_20260919/model_33366.pt"
