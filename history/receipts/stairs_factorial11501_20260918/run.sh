#!/usr/bin/env bash
set -uo pipefail
R=/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917
OUT="$R/recording/stairs_factorial11501_20260918"
PY=/root/autodl-tmp/conda/envs/isaaclab/bin/python
cd "$R"
for randomization in nominal full; do
  "$PY" "$OUT/diagnose_stairs.py" --root "$R" --policy "$R/runs/official4096_30000/student_11501.pt" --output "$OUT/direct_$randomization" --commands direct --randomization "$randomization" > "$OUT/direct_$randomization.log" 2>&1 &
  p1=$!
  "$PY" "$OUT/diagnose_stairs.py" --root "$R" --policy "$R/runs/official4096_30000/student_11501.pt" --output "$OUT/original_$randomization" --commands original --randomization "$randomization" > "$OUT/original_$randomization.log" 2>&1 &
  p2=$!
  echo "$p1 $p2" > "$OUT/current_pids.txt"
  wait "$p1"
  echo "direct_$randomization exit=$?"
  wait "$p2"
  echo "original_$randomization exit=$?"
done
