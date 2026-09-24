#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
export PYTHONUNBUFFERED=1
cd "$R/code_pit_posture_20260920_v1"
"$P" "$R/posture_probe.py" --root "$R" --code "$R/code_pit_posture_20260920_v1" --checkpoint "$R/runs/mixed9_4096_60000_uphill_speed_20260920/model_41000.pt" --output "$R/recording/pit_posture_frozen_20260920"
"$P" train.py --root "$R" --output "$R/runs/pit_posture_smoke_20260920" --num-envs 4096 --updates 8 --mixed-pit --pit-posture-profile phase_stance_v1 --resume "$R/runs/mixed9_4096_60000_uphill_speed_20260920/model_41000.pt"
"$P" train.py --root "$R" --output "$R/runs/pit_posture_inherit_20260920" --num-envs 4096 --updates 2 --mixed-pit --resume "$R/runs/pit_posture_smoke_20260920/model_41008.pt"
