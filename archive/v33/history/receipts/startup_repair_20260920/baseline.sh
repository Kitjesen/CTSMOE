#!/bin/bash
set -eu
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python startup_repair_audit.py --root "$PWD" --selection "$PWD/startup_repair_baseline_selection.json" --output "$PWD/recording/startup_repair_baseline_20260920"
