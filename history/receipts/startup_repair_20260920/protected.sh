#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R/code_startup_repair_20260920_v4"
export PYTHONUNBUFFERED=1
"$P" -m pytest -q test_startup_repair.py test_guard_math.py
"$P" train.py --root "$R" --output "$R/runs/startup_repair_protected_smoke_20260920" --num-envs 4096 --updates 8 --mixed-pit --startup-repair-profile omni_start_v4 --resume "$R/runs/startup_repair_repeat96_20260920/model_43105.pt"
"$P" "$R/startup_repair_verify_protected.py"
"$P" train.py --root "$R" --output "$R/runs/startup_repair_protected96_20260920" --num-envs 4096 --updates 96 --mixed-pit --startup-repair-profile omni_start_v4 --resume "$R/runs/startup_repair_repeat96_20260920/model_43105.pt"
"$P" "$R/startup_repair_audit.py" --root "$R" --selection "$R/startup_repair_protected_selection.json" --output "$R/recording/startup_repair_protected96_20260920"
"$P" "$R/startup_repair_regression.py" --root "$R" --selection "$R/startup_repair_protected_selection.json" --output "$R/recording/startup_repair_protected_regression96_20260920"
