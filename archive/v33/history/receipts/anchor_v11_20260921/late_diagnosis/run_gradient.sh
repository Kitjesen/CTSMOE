#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
S="$R/anchor_v11_stage_20260921/late_diagnosis"
O="$R/recording/anchor_v11_late_diagnosis_attempt2_20260921"
"$P" "$S/gradient_audit_v2.py" --selection "$S/selection.json" --traces-dir "$O" --checkpoint-root "$R" --upstream-source "$R/upstream/source/rsl_rl" --runner-config "$R/upstream/source/robot_lab/robot_lab/tasks/go2/rsl_rl_cfg.py" --code "$R/code_anchor_v11_20260921" --output "$O/gradient_audit.json" --device cuda:0
