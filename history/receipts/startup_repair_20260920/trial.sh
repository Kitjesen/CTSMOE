#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R/code_startup_repair_20260920_v1"
export PYTHONUNBUFFERED=1
exec "$P" train.py --root "$R" --output "$R/runs/startup_repair_trial96_20260920" --num-envs 4096 --updates 96 --mixed-pit --startup-repair-profile omni_start_v1 --resume "$R/runs/mixed9_4096_60000_strong_stance_video_20260920/model_42913.pt"
