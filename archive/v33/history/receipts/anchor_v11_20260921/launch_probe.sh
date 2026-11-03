#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R/code_flat_v10_20260921"
exec "$P" -u "$R/anchor_v11_stage_20260921/probe_evaluate.py" --root "$R" --code "$R/code_flat_v10_20260921" --selection "$R/anchor_v11_stage_20260921/probe_selection.json" --output "$R/recording/anchor_recovery_probe_20260921"
