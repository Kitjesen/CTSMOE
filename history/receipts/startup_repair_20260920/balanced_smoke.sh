#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
test ! -e "$R/code_startup_repair_20260920_v5"
cp -a "$R/code_startup_repair_20260920_v4" "$R/code_startup_repair_20260920_v5"
for name in startup_repair.py train.py test_startup_repair.py; do
  cp "$R/$name.balanced_pending" "$R/code_startup_repair_20260920_v5/$name"
done
cd "$R/code_startup_repair_20260920_v5"
export PYTHONUNBUFFERED=1
"$P" -m pytest -q test_startup_repair.py test_guard_math.py
"$P" train.py --root "$R" --output "$R/runs/startup_repair_balanced_smoke_20260920" --num-envs 4096 --updates 8 --mixed-pit --startup-repair-profile omni_start_v5 --resume "$R/runs/startup_repair_repeat96_20260920/model_43105.pt"
