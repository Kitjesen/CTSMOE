#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_flat_v9_20260921
export OMNI_KIT_ACCEPT_EULA=YES
nohup /root/autodl-tmp/conda/envs/isaaclab/bin/python -u /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/posture_cause_stage_20260921/reward_probe.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --code /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_flat_v9_20260921 --selection /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/flat_v9_selection_20260921.json --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/posture_cause_reward_20260921 >/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/posture_cause_reward_20260921.log 2>&1 </dev/null &
echo PROBE_PID=$!
