#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
stage="$root/startup_v13_stage_20260921"
lock="$root/full_episode_v12_pipeline.lock"
preflight="$root/recording/startup_v13_preflight_3000_20260921"
finalized="$preflight/FINALIZED.json"
log_path="$stage/run_pipeline.log"
lock_acquired=0

record_failure() {
  exit_code=$?
  if (( exit_code != 0 && lock_acquired == 1 )); then
    printf '{"state":"failed_before_training","updated_unix":%s,"exit_code":%s,"log_path":"%s","training_started":false}\n' \
      "$(date +%s)" "$exit_code" "$log_path" > "$stage/pipeline_status.tmp"
    mv "$stage/pipeline_status.tmp" "$stage/pipeline_status.json"
  fi
}
trap record_failure EXIT

mkdir -p "$stage"
exec 9>"$lock"
if ! flock -n 9; then
  echo 'REFUSED: another full-episode pipeline owns the shared lock' >&2
  exit 1
fi
lock_acquired=1
if [[ ! -f "$finalized" ]]; then
  printf '{"state":"refused_missing_preflight","updated_unix":%s,"training_started":false}\n' "$(date +%s)" \
    > "$stage/pipeline_status.tmp"
  mv "$stage/pipeline_status.tmp" "$stage/pipeline_status.json"
  echo 'REFUSED: V13 preflight FINALIZED.json is required before pipeline launch' >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'REFUSED: GPU already has an active compute process' >&2
  exit 1
fi

trap - EXIT
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python -u "$stage/run_pipeline.py"
