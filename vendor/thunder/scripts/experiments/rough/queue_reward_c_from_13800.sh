#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DOSO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON="${ISAAC_PYTHON:-python}"
TB_PYTHON="${TENSORBOARD_PYTHON:-$PYTHON}"
LAUNCHER="$SCRIPT_DIR/run_reward_c_from_13800.sh"
EVALUATOR="$SCRIPT_DIR/evaluate_checkpoint.sh"
RUN_ROOT="$ROOT/logs/rsl_rl/doso_thunder_terrain_inward_rough"
OUTPUT_ROOT="${ROUGH_REWARD_C_OUTPUT_ROOT:-$ROOT/outputs/thunder_rough_reward_c_13800}"
PROGRESS_LOG="$OUTPUT_ROOT/progress.log"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_FIXED_EVAL="${RUN_FIXED_EVAL:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
SMOKE_NAME="${SMOKE_RUN_NAME:-rough_reward_c_fs1_track5_3_up2_cmd10_from13800_smoke200}"
FULL_NAME="${FULL_RUN_NAME:-rough_reward_c_fs1_track5_3_up2_cmd10_from14000_cont1800}"

mkdir -p "$OUTPUT_ROOT/logs"
[[ -f "$LAUNCHER" && -f "$EVALUATOR" ]] || {
    printf 'rough launcher or evaluator is missing under %s\n' "$SCRIPT_DIR" >&2
    exit 2
}
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'invalid POLL_SECONDS: %s\n' "$POLL_SECONDS" >&2
    exit 2
}
[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ && "$RUN_FIXED_EVAL" =~ ^[01]$ ]] || {
    printf 'invalid NUM_ENVS or RUN_FIXED_EVAL\n' >&2
    exit 2
}

record() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$PROGRESS_LOG"
}

gpu_queue_busy() {
    pgrep -f '[s]cripts/reinforcement_learning/rsl_rl/(train|evaluate).*\.py' >/dev/null
}

latest_run() {
    find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*_$1" \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

latest_checkpoint() {
    find "$1" -maxdepth 1 -type f -name 'model_*.pt' -printf '%p\n' \
        | sort -V | tail -n 1
}

validate_smoke() {
    local run_dir="$1"
    "$TB_PYTHON" - "$run_dir" <<'PY'
import math
import sys
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = Path(sys.argv[1])
with (run_dir / "params" / "env.yaml").open(encoding="utf-8") as stream:
    cfg = yaml.load(stream, Loader=yaml.UnsafeLoader)

expected_rewards = {
    "undesired_contacts": 0.0,
    "contact_forces": 0.0,
    "touchdown_velocity_l2": 0.0,
    "feet_stumble": -1.0,
    "upward": 2.0,
    "track_lin_vel_xy_exp": 5.0,
    "track_ang_vel_z_exp": 3.0,
}
for name, expected in expected_rewards.items():
    actual = float(cfg["rewards"][name]["weight"])
    if actual != expected:
        raise SystemExit(f"reward mismatch: {name}={actual}, expected={expected}")

command_cfg = cfg["commands"]["base_velocity"]
ratio_names = (
    "rel_standing_envs",
    "rel_pure_yaw_envs",
    "rel_pure_lateral_envs",
)
for name in ratio_names:
    actual = float(command_cfg[name])
    if actual != 0.1:
        raise SystemExit(f"command ratio mismatch: {name}={actual}")
mixed = 1.0 - sum(float(command_cfg[name]) for name in ratio_names)
if not math.isclose(mixed, 0.7, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit(f"mixed command ratio mismatch: {mixed}")

acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
acc.Reload()
required = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Curriculum/terrain_levels",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Episode_Reward/upward",
    "Episode_Reward/feet_stumble",
    "Episode_Reward/lin_vel_z_l2",
    "Episode_Reward/ang_vel_xy_l2",
    "Episode_Reward/joint_pos_limits",
)
averages = {}
for tag in required:
    values = acc.Scalars(tag)
    if len(values) < 190:
        raise SystemExit(f"insufficient scalar samples: {tag}={len(values)}")
    tail = [float(item.value) for item in values[-50:]]
    if not all(math.isfinite(value) for value in tail):
        raise SystemExit(f"non-finite scalar: {tag}")
    averages[tag] = sum(tail) / len(tail)

limits = {
    "Train/mean_reward": (150.0, math.inf),
    "Train/mean_episode_length": (500.0, math.inf),
    "Curriculum/terrain_levels": (3.0, math.inf),
    "Metrics/base_velocity/error_vel_xy": (-math.inf, 1.5),
    "Metrics/base_velocity/error_vel_yaw": (-math.inf, 1.2),
    "Episode_Reward/upward": (5.0, math.inf),
    "Episode_Reward/feet_stumble": (-0.5, math.inf),
    "Episode_Reward/joint_pos_limits": (-0.2, math.inf),
}
for tag, (lower, upper) in limits.items():
    value = averages[tag]
    if not lower <= value <= upper:
        raise SystemExit(f"smoke gate failed: {tag}={value}, range=[{lower},{upper}]")

print("SMOKE_GATE_PASS", averages, flush=True)
PY
}

trap 'record "QUEUE_FAILED line=$LINENO exit=$?"' ERR
record "QUEUE_CREATED stages=200+1800 source=model_13800"

while gpu_queue_busy; do
    record "WAIT_GPU existing_training_or_evaluation"
    sleep "$POLL_SECONDS"
done

exec 9>"$ROOT/outputs/.gpu0_training.lock"
flock 9
while gpu_queue_busy; do
    record "WAIT_GPU race_after_lock"
    sleep "$POLL_SECONDS"
done

smoke_started="$(date +%s)"
record "TRAIN_SMOKE_START run=$SMOKE_NAME envs=$NUM_ENVS iterations=200"
MAX_ITERATIONS=200 NUM_ENVS="$NUM_ENVS" RUN_NAME="$SMOKE_NAME" bash "$LAUNCHER"
smoke_elapsed=$(( $(date +%s) - smoke_started ))
smoke_seconds_per_iteration="$(awk -v seconds="$smoke_elapsed" 'BEGIN { printf "%.3f", seconds / 200.0 }')"
record "TRAIN_SMOKE_COMPLETE run=$SMOKE_NAME seconds=$smoke_elapsed seconds_per_iteration=$smoke_seconds_per_iteration"

smoke_run="$(latest_run "$SMOKE_NAME")"
[[ -n "$smoke_run" ]] || { record "SMOKE_RUN_NOT_FOUND"; exit 2; }
smoke_checkpoint="$(latest_checkpoint "$smoke_run")"
[[ -f "$smoke_checkpoint" ]] || { record "SMOKE_CHECKPOINT_NOT_FOUND"; exit 2; }
validate_smoke "$smoke_run" 2>&1 | tee "$OUTPUT_ROOT/logs/smoke_gate.log"
record "TRAIN_SMOKE_GATE_PASS checkpoint=$smoke_checkpoint"

smoke_sha256="$(sha256sum "$smoke_checkpoint" | awk '{print $1}')"
full_started="$(date +%s)"
record "TRAIN_FULL_START run=$FULL_NAME envs=$NUM_ENVS iterations=1800"
SOURCE_RUN="$(basename "$smoke_run")" \
CHECKPOINT="$(basename "$smoke_checkpoint")" \
EXPECTED_SOURCE_SHA256="$smoke_sha256" \
MAX_ITERATIONS=1800 \
NUM_ENVS="$NUM_ENVS" \
RUN_NAME="$FULL_NAME" \
    bash "$LAUNCHER"
full_elapsed=$(( $(date +%s) - full_started ))
full_run="$(latest_run "$FULL_NAME")"
[[ -n "$full_run" ]] || { record "FULL_RUN_NOT_FOUND"; exit 2; }
full_checkpoint="$(latest_checkpoint "$full_run")"
[[ -f "$full_checkpoint" ]] || { record "FULL_CHECKPOINT_NOT_FOUND"; exit 2; }
full_checkpoint_bytes="$(stat -c '%s' "$full_checkpoint")"
full_checkpoint_sha256="$(sha256sum "$full_checkpoint" | awk '{print $1}')"
full_seconds_per_iteration="$(awk -v seconds="$full_elapsed" 'BEGIN { printf "%.3f", seconds / 1800.0 }')"
record "TRAIN_FULL_COMPLETE run=$FULL_NAME seconds=$full_elapsed seconds_per_iteration=$full_seconds_per_iteration checkpoint=$full_checkpoint checkpoint_bytes=$full_checkpoint_bytes checkpoint_sha256=$full_checkpoint_sha256"

if [[ "$RUN_FIXED_EVAL" == "1" ]]; then
    eval_dir="$OUTPUT_ROOT/fixed_eval/$(basename "$full_run")"
    record "FIXED_EVAL_START checkpoint=$full_checkpoint"
    CHECKPOINT="$full_checkpoint" OUT_DIR="$eval_dir" bash "$EVALUATOR"
    record "FIXED_EVAL_COMPLETE output=$eval_dir"
else
    record "FIXED_EVAL_SKIPPED RUN_FIXED_EVAL=$RUN_FIXED_EVAL"
fi
