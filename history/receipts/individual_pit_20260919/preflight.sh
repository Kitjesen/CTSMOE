#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R/code_individual_pit_20260919_v1"
export PYTHONUNBUFFERED=1
"$P" train.py --root "$R" --output "$R/runs/individual_pit_smoke_20260919" --num-envs 4096 --updates 32 --mixed-pit --pit-course-profile individual_speed_v1 --resume "$R/runs/mixed9_4096_60000_adaptive_20260919/model_34413.pt"
"$P" train.py --root "$R" --output "$R/runs/individual_pit_inherit_20260919" --num-envs 4096 --updates 2 --mixed-pit --resume "$R/runs/individual_pit_smoke_20260919/model_34445.pt"
