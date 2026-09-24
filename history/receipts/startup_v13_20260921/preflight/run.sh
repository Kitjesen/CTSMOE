#!/usr/bin/env bash
set -euo pipefail

R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
S="$R/startup_v13_stage_20260921/preflight"
CODE="$R/code_startup_v13_20260921"
CKPT="$R/runs/flat_v9_1000_20260921/model_44710.pt"
REF="$R/runs/startup_repair_flat40_96_20260920/model_43201.pt"
DONOR="$R/runs/anchor_v11_pilot_20260921/model_44758.pt"
SMOKE="$R/recording/startup_v13_preflight_smoke_20260921"
FULL="$R/recording/startup_v13_preflight_3000_20260921"
RUN_SMOKE=${RUN_SMOKE:-0}

exec 9>"$R/full_episode_v12_pipeline.lock"
flock -n 9 || { echo "shared full-episode pipeline lock is busy" >&2; exit 75; }
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "GPU process already active; preflight not started" >&2
  exit 76
fi
"$P" -m py_compile "$S/launcher.py" "$S/audit_stats.py" "$S/finalize_audit.py"
if [ "$RUN_SMOKE" = 1 ]; then
  timeout --signal=TERM --kill-after=30s 600 "$P" "$S/launcher.py" \
    --root "$R" --code "$CODE" --output "$SMOKE" --checkpoint "$CKPT" --reference "$REF" \
    --coefficient-donor "$DONOR" --smoke
fi
timeout --signal=TERM --kill-after=30s 3600 "$P" "$S/launcher.py" \
  --root "$R" --code "$CODE" --output "$FULL" --checkpoint "$CKPT" --reference "$REF" \
  --coefficient-donor "$DONOR"
"$P" "$S/finalize_audit.py" --output "$FULL" --checkpoint "$CKPT" --reference "$REF" \
  --coefficient-donor "$DONOR"
