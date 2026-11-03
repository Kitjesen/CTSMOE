#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DOSO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON="${ISAAC_PYTHON:-python}"
TRAIN="$ROOT/scripts/reinforcement_learning/rsl_rl/train.py"
TASK="DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0"
RUN_ROOT="$ROOT/logs/rsl_rl/doso_thunder_terrain_inward_rough"
DEFAULT_SOURCE_RUN="2026-09-02_05-32-28_rough_v0_4k_stumble_n01"
DEFAULT_CHECKPOINT="model_13800.pt"
DEFAULT_SOURCE_SHA256="b79c66749f5e1147e1c780c0ba7a35790c212b28f8789012e6fb97c8097e0727"
SOURCE_RUN="${SOURCE_RUN:-$DEFAULT_SOURCE_RUN}"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-}"
NUM_ENVS="${NUM_ENVS:-4096}"
DEVICE="${DEVICE:-cuda:0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
OUTPUT_ROOT="${ROUGH_AB_OUTPUT_ROOT:-$ROOT/outputs/thunder_rough_ab_13800}"
STARTED_AT="$(date '+%Y%m%d_%H%M%S')"

if [[ -z "$EXPECTED_SOURCE_SHA256" && "$SOURCE_RUN" == "$DEFAULT_SOURCE_RUN" && "$CHECKPOINT" == "$DEFAULT_CHECKPOINT" ]]; then
    EXPECTED_SOURCE_SHA256="$DEFAULT_SOURCE_SHA256"
fi
[[ "$EXPECTED_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'EXPECTED_SOURCE_SHA256 is required for a non-default parent checkpoint\n' >&2
    exit 2
}
[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ && "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'NUM_ENVS and MAX_ITERATIONS must be positive integers\n' >&2
    exit 2
}
SOURCE_CHECKPOINT="$RUN_ROOT/$SOURCE_RUN/$CHECKPOINT"
[[ -f "$SOURCE_CHECKPOINT" ]] || {
    printf 'source checkpoint not found: %s\n' "$SOURCE_CHECKPOINT" >&2
    exit 2
}
ACTUAL_SOURCE_SHA256="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
[[ "$ACTUAL_SOURCE_SHA256" == "$EXPECTED_SOURCE_SHA256" ]] || {
    printf 'source checkpoint SHA256 mismatch: expected=%s actual=%s\n' \
        "$EXPECTED_SOURCE_SHA256" "$ACTUAL_SOURCE_SHA256" >&2
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
    printf 'source_run=%s\n' "$SOURCE_RUN"
    printf 'checkpoint=%s\n' "$CHECKPOINT"
    printf 'checkpoint_bytes=%s\n' "$(wc -c <"$SOURCE_CHECKPOINT")"
    printf 'checkpoint_sha256=%s\n' "$ACTUAL_SOURCE_SHA256"
    printf 'num_envs=%s\n' "$NUM_ENVS"
    printf 'max_iterations=%s\n' "$MAX_ITERATIONS"
} >"$OUTPUT_ROOT/manifests/command_sampling_ab_${STARTED_AT}.txt"

run_branch() {
    local run_name="$1"
    local standing="$2"
    local yaw="$3"
    local lateral="$4"

    "$PYTHON" "$TRAIN" \
        --task "$TASK" \
        --num_envs "$NUM_ENVS" \
        --seed 42 \
        --headless \
        --device "$DEVICE" \
        --max_iterations "$MAX_ITERATIONS" \
        --run_name "$run_name" \
        --resume \
        --load_run "$SOURCE_RUN" \
        --checkpoint "$CHECKPOINT" \
        env.rewards.feet_stumble.weight=-0.1 \
        env.commands.base_velocity.rel_standing_envs="$standing" \
        env.commands.base_velocity.rel_pure_yaw_envs="$yaw" \
        env.commands.base_velocity.rel_pure_lateral_envs="$lateral" \
        2>&1 | tee "$OUTPUT_ROOT/logs/${run_name}_${STARTED_AT}.log"
}

# A = 10/10/10/70; B = 5/5/5/85. The mixed share is the implicit remainder.
run_branch rough_ab_a_ctrl_10_10_10_70_from13800 0.10 0.10 0.10
run_branch rough_ab_b_low_05_05_05_85_from13800 0.05 0.05 0.05
