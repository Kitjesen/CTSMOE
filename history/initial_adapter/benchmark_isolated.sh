#!/bin/bash
set -euo pipefail
root=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
output=${1:-benchmark_delay4096}
if [ "$#" -gt 0 ]; then shift; fi
active_pid=12542
command_line=$(tr '\0' ' ' < /proc/$active_pid/cmdline)
[[ "$command_line" == *"$root/runs/official4096_30000"* ]]
trap 'kill -CONT "$active_pid" 2>/dev/null || true' EXIT
kill -STOP "$active_pid"
printf 'PAUSED_PID=%s\n' "$active_pid"
export PYTHONUNBUFFERED=1
cd "$root/delay_optimization"
timeout --kill-after=30s 900s /root/autodl-tmp/conda/envs/isaaclab/bin/python \
  benchmark_training_delay.py --root "$root" \
  --output "$root/runs/$output" \
  --resume "$root/runs/official4096_30000/model_7000.pt" "$@"
