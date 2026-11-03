#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
PY=/root/autodl-tmp/conda/envs/isaaclab/bin/python
STAGE="$ROOT/full_episode_v12_stage_20260921/gradient_diagnosis"
exec 9>"$ROOT/full_episode_v12_stage_20260921/gradient_diagnosis.lock"
flock -n 9 || exit 73
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo 'GPU already has a compute process; diagnosis not started.' >&2
  exit 74
fi
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
cd "$ROOT"
exec "$PY" -u "$STAGE/launcher.py" --root "$ROOT" --code "$ROOT/code_full_episode_v12_20260921" --output "$ROOT/recording/full_episode_v12_gradient_diagnosis_20260921" --checkpoint "$ROOT/runs/full_episode_v12_pilot300_20260921/model_45010.pt" --reference "$ROOT/runs/startup_repair_flat40_96_20260920/model_43201.pt"
