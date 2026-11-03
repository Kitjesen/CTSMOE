#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
stage="$root/full_episode_v12_stage_20260921"
lock="$root/full_episode_v12_pipeline.lock"
preflight="$root/recording/full_episode_v12_preflight_3000_20260921"
finalized="$preflight/FINALIZED.json"
audit_status="$preflight/audit_status.json"
log_path="$stage/run_pipeline.log"
lock_acquired=0

record_early_failure() {
  exit_code=$?
  if (( exit_code != 0 && lock_acquired == 1 )); then
    now=$(date +%s)
    printf '{"state":"failed_before_training","updated_unix":%s,"exit_code":%s,"log_path":"%s","training_started":false}\n' \
      "$now" "$exit_code" "$log_path" > "$stage/pipeline_status.tmp"
    mv "$stage/pipeline_status.tmp" "$stage/pipeline_status.json"
  fi
}
trap record_early_failure EXIT

mkdir -p "$stage"
exec 9>"$lock"
if ! flock -n 9; then
  echo 'REFUSED: another V12 pipeline owns the global lock' >&2
  exit 1
fi
lock_acquired=1
report_launcher_exit() {
  local result=$?
  if (( result != 0 )); then
    printf '{"state":"failed_before_training","updated_unix":%s,"exit_code":%s,"log":"pipeline.log","training_started":false}\n' \
      "$(date +%s)" "$result" > "$stage/pipeline_status.tmp"
    mv "$stage/pipeline_status.tmp" "$stage/pipeline_status.json"
  fi
}
trap report_launcher_exit EXIT
if [[ ! -f "$finalized" ]]; then
  now=$(date +%s)
  printf '{"state":"waiting_for_preflight","updated_unix":%s,"training_started":false}\n' "$now" \
    > "$stage/pipeline_status.tmp"
  mv "$stage/pipeline_status.tmp" "$stage/pipeline_status.json"
  deadline=$((SECONDS + 3600))
  while [[ ! -f "$finalized" ]]; do
    if [[ -f "$audit_status" ]] && grep -Eq '"state"[[:space:]]*:[[:space:]]*"AUDIT_FAILED"' "$audit_status"; then
      echo 'REFUSED: V12 preflight reported AUDIT_FAILED' >&2
      exit 1
    fi
    if ! pgrep -af '[f]ull_episode_v12.*(preflight.*run\.sh|launcher\.py)' >/dev/null; then
      echo 'REFUSED: V12 preflight ended without FINALIZED.json' >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo 'REFUSED: V12 preflight wait exceeded one hour' >&2
      exit 1
    fi
    sleep 20
  done
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'REFUSED: GPU already has an active compute process' >&2
  exit 1
fi

trap - EXIT
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python -u "$stage/run_pipeline.py"
