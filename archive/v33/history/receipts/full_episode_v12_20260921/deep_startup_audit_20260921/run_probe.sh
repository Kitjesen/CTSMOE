#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
stage="$root/deep_startup_audit_20260921"
exec 9>"$root/full_episode_v12_pipeline.lock"
flock -n 9
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'REFUSED: GPU already occupied' >&2
  exit 1
fi
export PYTHONPATH="$root/full_episode_v12_stage_20260921:$root/code_full_episode_v12_20260921"
export OMNI_KIT_ACCEPT_EULA=YES
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python -u "$stage/probe_r2.py" --root "$root" --code "$root/code_full_episode_v12_20260921" --selection "$stage/selection.json" --output "$root/recording/deep_startup_audit_20260921_r2" --flat25
