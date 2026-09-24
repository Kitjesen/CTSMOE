#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_flat_v10_20260921
export OMNI_KIT_ACCEPT_EULA=YES
nohup /root/autodl-tmp/conda/envs/isaaclab/bin/python -u /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/flat_v10_stage_20260921/run_pipeline.py >/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/flat_v10_1000_20260921.log 2>&1 </dev/null &
echo PIPELINE_PID=$!
