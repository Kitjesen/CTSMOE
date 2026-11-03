#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_flat_v9_20260921
export OMNI_KIT_ACCEPT_EULA=YES
nohup /root/autodl-tmp/conda/envs/isaaclab/bin/python -u /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/flat_v9_stage_20260921/evaluate_v9.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --code /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_flat_v9_20260921 --selection /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/flat_v9_stage_20260921/eval_smoke_selection.json --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/flat_v9_eval_smoke_20260921 --smoke >/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/flat_v9_eval_smoke_20260921.log 2>&1 </dev/null &
echo EVAL_SMOKE_PID=$!
