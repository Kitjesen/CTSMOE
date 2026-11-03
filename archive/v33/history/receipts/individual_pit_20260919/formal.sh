#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
cd "$R/code_individual_pit_20260919_v1"
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python train.py --root "$R" --output "$R/runs/mixed9_4096_60000_individualpit_20260919" --num-envs 4096 --updates 25552 --mixed-pit --resume "$R/runs/individual_pit_inherit_20260919/model_34447.pt"
