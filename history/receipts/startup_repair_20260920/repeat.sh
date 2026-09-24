#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R/code_startup_repair_20260920_v2"
export PYTHONUNBUFFERED=1
"$P" -m pytest -q test_startup_repair.py test_terrain_allocation.py test_surface_posture.py
"$P" train.py --root "$R" --output "$R/runs/startup_repair_repeat_smoke_20260920" --num-envs 4096 --updates 8 --mixed-pit --startup-repair-profile omni_start_v2 --resume "$R/runs/startup_repair_trial96_20260920/model_43009.pt"
"$P" "$R/startup_repair_verify_repeat.py"
"$P" train.py --root "$R" --output "$R/runs/startup_repair_repeat96_20260920" --num-envs 4096 --updates 96 --mixed-pit --startup-repair-profile omni_start_v2 --resume "$R/runs/startup_repair_trial96_20260920/model_43009.pt"
"$P" "$R/startup_repair_audit.py" --root "$R" --selection "$R/startup_repair_repeat_selection.json" --output "$R/recording/startup_repair_repeat96_20260920"
"$P" "$R/startup_repair_regression.py" --root "$R" --selection "$R/startup_repair_repeat_selection.json" --output "$R/recording/startup_repair_repeat_regression96_20260920"
