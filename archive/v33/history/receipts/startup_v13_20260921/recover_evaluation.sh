#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/root/cts-v13-evaluation-recovery-20260921/tmp
mkdir -p "$TMPDIR"
exec /root/autodl-tmp/conda/envs/isaaclab/bin/python -u /root/cts-v13-evaluation-recovery-20260921/recover_evaluation.py
