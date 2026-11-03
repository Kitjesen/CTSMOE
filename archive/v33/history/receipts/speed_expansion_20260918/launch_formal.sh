#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/code
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python /root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/speed_expansion_20260918/resume_formal.py
