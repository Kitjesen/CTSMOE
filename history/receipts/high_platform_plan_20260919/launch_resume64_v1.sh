#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_pit_20260919_v1
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python train_pit.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/pit_resume64_20260919_v1 --num-envs 64 --updates 2 --save-interval 2 --max-level 4 --resume /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/pit_smoke64_20260919_v1/model_3.pt
