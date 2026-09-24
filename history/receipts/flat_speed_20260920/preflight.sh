#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R/code_flat_speed_20260920_v1"
export PYTHONUNBUFFERED=1
"$P" train.py --root "$R" --output "$R/runs/flat_speed_smoke_20260920" --num-envs 4096 --updates 8 --mixed-pit --expand-flat-speed --resume "$R/runs/mixed9_4096_60000_individualpit_20260919/model_37000.pt"
"$P" train.py --root "$R" --output "$R/runs/flat_speed_inherit_20260920" --num-envs 4096 --updates 2 --mixed-pit --resume "$R/runs/flat_speed_smoke_20260920/model_37008.pt"
