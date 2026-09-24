#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_flat_v9_20260921
export OMNI_KIT_ACCEPT_EULA=YES
nohup /root/autodl-tmp/conda/envs/isaaclab/bin/python -u /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/flat_v9_stage_20260921/continue_and_evaluate.py >/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/flat_v9_1000_20260921.log 2>&1 </dev/null &
echo PIPELINE_PID=$!
