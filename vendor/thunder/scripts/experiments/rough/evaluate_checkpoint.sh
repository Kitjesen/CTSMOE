#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DOSO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON="${ISAAC_PYTHON:-python}"
EVALUATOR="$ROOT/scripts/reinforcement_learning/rsl_rl/evaluate_rough.py"
DEFAULT_CHECKPOINT="$ROOT/logs/rsl_rl/doso_thunder_terrain_inward_rough/2026-09-02_05-32-28_rough_v0_4k_stumble_n01/model_13800.pt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
DEVICE="${DEVICE:-cuda:0}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
ENVS_PER_SCENARIO="${ENVS_PER_SCENARIO:-16}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/thunder_rough_eval/$(basename "$CHECKPOINT" .pt)}"

[[ -f "$EVALUATOR" ]] || {
    printf 'rough evaluator not found: %s\n' "$EVALUATOR" >&2
    exit 2
}
[[ -f "$CHECKPOINT" ]] || {
    printf 'checkpoint not found: %s\n' "$CHECKPOINT" >&2
    exit 2
}
[[ "$ENVS_PER_SCENARIO" =~ ^[1-9][0-9]*$ ]] || {
    printf 'invalid ENVS_PER_SCENARIO: %s\n' "$ENVS_PER_SCENARIO" >&2
    exit 2
}

mkdir -p "$OUT_DIR"
cd "$ROOT"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"

run_case() {
    local label="$1"
    local terrain="$2"
    local suite="$3"
    local steps="$4"

    "$PYTHON" "$EVALUATOR" \
        --checkpoint "$CHECKPOINT" \
        --terrain "$terrain" \
        --difficulty 0.5 \
        --suite "$suite" \
        --envs_per_scenario "$ENVS_PER_SCENARIO" \
        --steps "$steps" \
        --warmup_steps 50 \
        --device "$DEVICE" \
        --policy_device "$POLICY_DEVICE" \
        --output "$OUT_DIR/${label}.json" \
        2>&1 | tee "$OUT_DIR/${label}.log"
}

run_case stairs_down_d05 pyramid_stairs stairs 350
run_case stairs_up_d05 pyramid_stairs_inv stairs 350
run_case motions_random_rough_d05 random_rough motions 400
printf 'ROUGH_EVAL_COMPLETE output=%s\n' "$OUT_DIR"
