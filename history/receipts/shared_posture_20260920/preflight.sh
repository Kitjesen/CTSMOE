#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
export PYTHONUNBUFFERED=1
cd "$R/code_shared_posture_20260920_v1"
"$P" -m pytest -q test_surface_posture.py test_pit_posture.py test_pit_climb.py test_pit_individual.py test_uphill_speed.py test_flat_speed.py test_terrain_allocation.py
"$P" train.py --root "$R" --output "$R/runs/shared_posture_smoke_20260920" --num-envs 4096 --updates 8 --mixed-pit --surface-posture-profile level_surface_v1 --resume "$R/runs/mixed9_4096_60000_pit_posture_20260920/model_41500.pt"
"$P" train.py --root "$R" --output "$R/runs/shared_posture_inherit_20260920" --num-envs 4096 --updates 2 --mixed-pit --resume "$R/runs/shared_posture_smoke_20260920/model_41508.pt"
