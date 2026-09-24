#!/bin/bash
set -euo pipefail
ROOT=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
cd "$ROOT/code_phase_v8_20260921"
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python -u train.py \
  --root "$ROOT" --output "$ROOT/runs/phase_v8_smoke_20260921" \
  --num-envs 4096 --updates 16 --mixed-pit --startup-repair-profile omni_start_v8 \
  --resume "$ROOT/runs/startup_repair_flat40_96_20260920/model_43201.pt"
