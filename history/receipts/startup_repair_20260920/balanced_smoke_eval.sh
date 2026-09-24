#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
export PYTHONUNBUFFERED=1
"$P" "$R/startup_repair_audit.py" --root "$R" --selection "$R/startup_balanced_smoke_selection.json" --output "$R/recording/startup_balanced_smoke_eval_20260920"
"$P" "$R/startup_repair_regression.py" --root "$R" --selection "$R/startup_balanced_smoke_selection.json" --output "$R/recording/startup_balanced_smoke_regression_20260920"
