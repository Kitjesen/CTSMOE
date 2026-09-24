#!/bin/bash
set -eu
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python posture_history_fine_probe.py --root "$PWD" --output "$PWD/recording/posture_history_fine_20260920"
