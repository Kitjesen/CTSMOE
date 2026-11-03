#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_relax45_20260919_v1
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python train.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_4096_60000_relax45_20260919 --num-envs 4096 --updates 29654 --mixed-pit --actuator-profile rs0402_transient_v4 --slope-profile slope45_v1 --resume /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_4096_60000_20260919/model_30345.pt

