#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_mixed9_20260919_v3
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python train.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_resume4096_v3_20260919 --num-envs 4096 --updates 8 --mixed-pit --resume /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_smoke4096_v2_20260919/model_30003.pt


