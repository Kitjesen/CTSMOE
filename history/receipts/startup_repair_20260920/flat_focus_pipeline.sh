#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
test ! -e "$R/code_startup_repair_20260920_v6"
cp -a "$R/code_startup_repair_20260920_v5" "$R/code_startup_repair_20260920_v6"
for name in startup_repair.py terrain_allocation.py train.py test_terrain_allocation.py; do
  cp "$R/$name.flat40_pending" "$R/code_startup_repair_20260920_v6/$name"
done
cd "$R/code_startup_repair_20260920_v6"
export PYTHONUNBUFFERED=1
"$P" -m pytest -q test_startup_repair.py test_terrain_allocation.py test_guard_math.py
"$P" train.py --root "$R" --output "$R/runs/startup_repair_flat40_96_20260920" --num-envs 4096 --updates 96 --mixed-pit --startup-repair-profile omni_start_v6 --resume "$R/runs/startup_repair_repeat96_20260920/model_43105.pt"
"$P" "$R/startup_repair_audit.py" --root "$R" --selection "$R/startup_flat_focus_selection.json" --output "$R/recording/startup_flat40_eval_20260920"
"$P" "$R/startup_repair_regression.py" --root "$R" --selection "$R/startup_flat_focus_selection.json" --output "$R/recording/startup_flat40_regression_20260920"
echo FLAT_FOCUS_PIPELINE_COMPLETE
