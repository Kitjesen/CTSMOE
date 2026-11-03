#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DOSO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON="${ISAAC_PYTHON:-python}"
TRAIN="$ROOT/scripts/reinforcement_learning/rsl_rl/train.py"
TASK="DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0"
NUM_ENVS="${NUM_ENVS:-4096}"
DEVICE="${DEVICE:-cuda:0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
RUN_NAME="${RUN_NAME:-rough_v0_4k_stumble_n01}"
FEET_STUMBLE_WEIGHT="${FEET_STUMBLE_WEIGHT:--0.1}"
OUTPUT_ROOT="${ROUGH_OUTPUT_ROOT:-$ROOT/outputs/thunder_rough_v0}"
STARTED_AT="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$OUTPUT_ROOT/logs/${RUN_NAME}_${STARTED_AT}.log"
MANIFEST_FILE="$OUTPUT_ROOT/manifests/${RUN_NAME}_${STARTED_AT}.txt"

[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'invalid NUM_ENVS: %s\n' "$NUM_ENVS" >&2
    exit 2
}
[[ "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'invalid MAX_ITERATIONS: %s\n' "$MAX_ITERATIONS" >&2
    exit 2
}
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf 'invalid RUN_NAME: %s\n' "$RUN_NAME" >&2
    exit 2
}
[[ -f "$TRAIN" ]] || {
    printf 'training entry point not found: %s\n' "$TRAIN" >&2
    exit 2
}

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/manifests"
cd "$ROOT"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"

git_commit="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
if [[ -n "$(git status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
    git_dirty=true
else
    git_dirty=false
fi
{
    printf 'task=%s\n' "$TASK"
    printf 'git_commit=%s\n' "$git_commit"
    printf 'git_dirty=%s\n' "$git_dirty"
    printf 'num_envs=%s\n' "$NUM_ENVS"
    printf 'device=%s\n' "$DEVICE"
    printf 'max_iterations=%s\n' "$MAX_ITERATIONS"
    printf 'run_name=%s\n' "$RUN_NAME"
    printf 'feet_stumble_weight=%s\n' "$FEET_STUMBLE_WEIGHT"
    "$PYTHON" --version
} >"$MANIFEST_FILE"

# This reproduces the model_13800 lineage: random initialization, 4096 envs,
# seed 42, and the single feet_stumble=-0.1 runtime override.
"$PYTHON" "$TRAIN" \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --seed 42 \
    --headless \
    --device "$DEVICE" \
    --max_iterations "$MAX_ITERATIONS" \
    --run_name "$RUN_NAME" \
    env.rewards.feet_stumble.weight="$FEET_STUMBLE_WEIGHT" \
    2>&1 | tee "$LOG_FILE"
