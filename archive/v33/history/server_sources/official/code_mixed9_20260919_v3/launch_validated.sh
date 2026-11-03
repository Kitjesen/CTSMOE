#!/bin/bash
set -euo pipefail
while kill -0 12148 2>/dev/null; do
  state=$(ps -p 12148 -o stat= || true)
  if [[ "$state" == Z* ]]; then break; fi
  sleep 2
done
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code
export PYTHONUNBUFFERED=1
/root/autodl-tmp/conda/envs/isaaclab/bin/python train.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/savecheck64 --num-envs 64 --updates 2 > /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/savecheck64.log 2>&1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python train.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/official4096_30000 --num-envs 4096 --updates 30000
