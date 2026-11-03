#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
bash "$R/startup_repair_trial.sh"
"$P" "$R/startup_repair_audit.py" --root "$R" --selection "$R/startup_repair_candidate_selection.json" --output "$R/recording/startup_repair_candidate96_20260920"
"$P" "$R/startup_repair_regression.py" --root "$R" --selection "$R/startup_repair_candidate_selection.json" --output "$R/recording/startup_repair_regression96_20260920"
