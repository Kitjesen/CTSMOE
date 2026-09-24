#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
cd "$R/code_anchor_v11_20260921"
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python -u "$R/anchor_v11_stage_20260921/run_pipeline.py"
