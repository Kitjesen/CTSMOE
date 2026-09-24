#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
S="$R/distribution_audit_stage_20260921"
exec 9>"$S/audit.lock"
flock -n 9 || exit 75
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "GPU process already active; audit not started"
  exit 76
fi
"$P" -m py_compile "$S/launcher.py" "$S/audit_stats.py"
timeout --signal=TERM --kill-after=30s 600 "$P" "$S/launcher.py" --root "$R" --code "$R/code_anchor_v11_20260921" --output "$R/recording/distribution_audit_smoke_20260921" --checkpoint "$R/runs/anchor_v11_pilot_20260921/model_44758.pt" --reference "$R/runs/startup_repair_flat40_96_20260920/model_43201.pt" --smoke
timeout --signal=TERM --kill-after=30s 1200 "$P" "$S/launcher.py" --root "$R" --code "$R/code_anchor_v11_20260921" --output "$R/recording/distribution_audit_480_20260921" --checkpoint "$R/runs/anchor_v11_pilot_20260921/model_44758.pt" --reference "$R/runs/startup_repair_flat40_96_20260920/model_43201.pt"
