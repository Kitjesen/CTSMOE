#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_climbfirst_20260919_v1
export PYTHONUNBUFFERED=1
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
"$P" train.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/climbfirst_smoke4096_20260919 --num-envs 4096 --updates 32 --mixed-pit --pit-training-profile climb_first_v2 --resume /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_4096_60000_relax45_20260919/model_31830.pt
"$P" train.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/climbfirst_inherit4096_20260919 --num-envs 4096 --updates 2 --mixed-pit --resume /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/climbfirst_smoke4096_20260919/model_31862.pt

