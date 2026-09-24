#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
S="$R/anchor_v11_stage_20260921"
C="$R/code_anchor_v11_20260921"
P=/root/autodl-tmp/conda/envs/isaaclab/bin/python
test ! -e "$C"
cp -a "$R/code_flat_v10_20260921" "$C"
cp "$S/anchor_cts_v2.py" "$S/anchor_cts.py"
cp "$S/anchor_cts.py" "$S/anchor_objective.py" "$S/train.py" "$C/"
export PYTHONPATH="$C:$R/upstream/source/rsl_rl"
"$P" -m pytest "$S/test_anchor_objective.py" "$S/test_anchor_cts_integration_v2.py" -q
