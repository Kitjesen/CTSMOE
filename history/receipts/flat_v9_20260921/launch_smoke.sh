#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_flat_v9_20260921
export OMNI_KIT_ACCEPT_EULA=YES
nohup /root/autodl-tmp/conda/envs/isaaclab/bin/python -u train.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/flat_v9_smoke_20260921 --num-envs 4096 --updates 16 --mixed-pit --startup-repair-profile omni_start_v9 --resume /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/phase_v8_1000_20260921/model_43710.pt >/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/flat_v9_smoke_20260921.log 2>&1 </dev/null &
echo SMOKE_PID=$!
