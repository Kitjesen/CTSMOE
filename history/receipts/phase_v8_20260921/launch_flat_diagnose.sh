#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_phase_v8_20260921
export OMNI_KIT_ACCEPT_EULA=YES
nohup /root/autodl-tmp/conda/envs/isaaclab/bin/python -u /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/phase_v8_stage_20260921/diagnose_flat_43501.py --root /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917 --code /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code_phase_v8_20260921 --selection /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/phase_v8_stage_20260921/diagnose_flat_selection.json --output /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/v8_flat_diagnose_43501 >/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/v8_flat_diagnose_43501.log 2>&1 </dev/null &
echo RECORD_PID=$!
