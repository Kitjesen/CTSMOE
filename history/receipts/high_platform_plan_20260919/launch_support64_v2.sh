#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_pit_20260919_v2
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python train_pit.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/pit_support64_v2_20260919 --num-envs 64 --support-check --max-level 21 --warm-start /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/official4096_30000_wave_surface_20260919/model_27500.pt
