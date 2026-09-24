#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
S="$R/anchor_v11_stage_20260921/late_diagnosis"
O="$R/recording/anchor_v11_late_diagnosis_20260921"
exec 9>"$S/evaluation.lock"
flock -n 9 || exit 75
if [ -e "$O" ]; then
  echo "Refusing to overwrite existing diagnosis"
  exit 2
fi
"$P" "$S/probe.py" --root "$R" --code "$R/code_anchor_v11_20260921" --selection "$S/selection.json" --output "$O"
"$P" "$S/analyze_probe.py" "$O"
