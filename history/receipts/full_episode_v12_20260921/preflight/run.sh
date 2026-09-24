#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
S="$R/full_episode_v12_stage_20260921/preflight"
CODE="$R/code_full_episode_v12_20260921"
CKPT="$R/runs/flat_v9_1000_20260921/model_44710.pt"
REF="$R/runs/startup_repair_flat40_96_20260920/model_43201.pt"
DONOR="$R/runs/anchor_v11_pilot_20260921/model_44758.pt"
SMOKE="$R/recording/full_episode_v12_preflight_smoke_20260921"
FULL="$R/recording/full_episode_v12_preflight_3000_20260921"
exec 9>"$S/preflight.lock"
flock -n 9 || exit 75
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "GPU process already active; preflight not started"
  exit 76
fi
"$P" -m py_compile "$S/launcher.py" "$S/audit_stats.py" "$S/finalize_audit.py"
timeout --signal=TERM --kill-after=30s 600 "$P" "$S/launcher.py" \
  --root "$R" --code "$CODE" --output "$SMOKE" --checkpoint "$CKPT" --reference "$REF" \
  --coefficient-donor "$DONOR" --smoke
timeout --signal=TERM --kill-after=30s 3600 "$P" "$S/launcher.py" \
  --root "$R" --code "$CODE" --output "$FULL" --checkpoint "$CKPT" --reference "$REF" \
  --coefficient-donor "$DONOR"
"$P" "$S/finalize_audit.py" --output "$FULL" --checkpoint "$CKPT" --reference "$REF" \
  --coefficient-donor "$DONOR"
