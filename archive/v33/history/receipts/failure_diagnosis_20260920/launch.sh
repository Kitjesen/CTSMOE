#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
export PYTHONUNBUFFERED=1
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python "$R/probe_wave_slopes_20260920.py" --root "$R" --code "$R/code_individual_pit_20260919_v1" --checkpoint "$R/runs/mixed9_4096_60000_individualpit_20260919/model_35500.pt" --output "$R/runs/wave_slopes_diagnosis_20260920"
