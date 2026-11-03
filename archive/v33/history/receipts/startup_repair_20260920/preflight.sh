#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R/code_startup_repair_20260920_v1"
export PYTHONUNBUFFERED=1
for name in startup_repair.py test_startup_repair.py terrain_allocation.py terrain_allocation_runtime.py train.py; do
    cp "$name.startup_pending" "$name"
done
"$P" -m pytest -q test_startup_repair.py test_terrain_allocation.py test_stance_strength.py test_surface_posture.py test_pit_posture.py test_pit_individual.py
"$P" train.py --root "$R" --output "$R/runs/startup_repair_smoke_20260920" --num-envs 4096 --updates 8 --mixed-pit --startup-repair-profile omni_start_v1 --resume "$R/runs/mixed9_4096_60000_strong_stance_video_20260920/model_42913.pt"
"$P" train.py --root "$R" --output "$R/runs/startup_repair_inherit_20260920" --num-envs 4096 --updates 2 --mixed-pit --resume "$R/runs/startup_repair_smoke_20260920/model_42921.pt"
