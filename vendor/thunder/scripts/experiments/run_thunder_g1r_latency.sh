#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${DOSO_ROOT:-/root/autodl-tmp/workspace/doso-train}"
PYTHON="${ISAAC_PYTHON:-/root/autodl-tmp/conda/envs/isaaclab/bin/python}"
TASK="DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0"
TRAIN="$ROOT/scripts/reinforcement_learning/rsl_rl/train.py"
EVALUATE="$ROOT/scripts/reinforcement_learning/rsl_rl/evaluate_torque_tracking.py"
GATE="$ROOT/scripts/experiments/g1r_result_gate.py"
RUN_ROOT="$ROOT/logs/rsl_rl/doso_thunder_terrain_inward_flat"
OUTPUT="${G1R_OUTPUT:-$ROOT/outputs/thunder_g1r_latency_20260902}"
G1_SOURCE="${G1_SOURCE:-$RUN_ROOT/2026-09-01_23-06-48_lateral_smooth_s4_7996_g1_v2/model_999.pt}"
YAW_SOURCE="${YAW_SOURCE:-$RUN_ROOT/2026-09-01_23-27-27_lateral_smooth_s4_7996_g2_v2/model_4998.pt}"
NOMINAL_WINNER="${NOMINAL_WINNER:-}"
NOMINAL_WINNER_FILE="${NOMINAL_WINNER_FILE:-$OUTPUT/nominal_winner.txt}"
MODE="${1:-help}"

case "$MODE" in
    yaw-specialization|yaw-soft-touchdown|symmetric-recovery|orientation-recovery|latency-probes|latency-combined) ;;
    *)
        printf 'Usage: %s {yaw-specialization|yaw-soft-touchdown|symmetric-recovery|orientation-recovery|latency-probes|latency-combined}\n' "$0"
        exit 2
        ;;
esac

mkdir -p "$OUTPUT/logs" "$OUTPUT/results" "$OUTPUT/manifests"
cd "$ROOT"
export ACCEPT_EULA=Y

record() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$OUTPUT/progress.log"; }
latest_checkpoint() { find "$1" -maxdepth 1 -type f -name 'model_*.pt' -printf '%p\n' | sort -V | tail -n 1; }
latest_run() { find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*_$1" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-; }
latest_run_any() {
    local pattern
    for pattern in "$@"; do
        find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n'
    done | sort -nr | head -n 1 | cut -d' ' -f2-
}

COMMON=(
    --task "$TASK" --num_envs 4096 --seed 42 --headless --device cuda:0
    env.scene.robot.actuators.hip.stiffness=90.0
    env.scene.robot.actuators.hip.damping=6.93
    env.scene.robot.actuators.thigh.stiffness=90.0
    env.scene.robot.actuators.thigh.damping=6.93
    env.scene.robot.actuators.calf.stiffness=90.0
    env.scene.robot.actuators.calf.damping=6.93
    env.commands.base_velocity.rel_standing_envs=0.10
    env.commands.base_velocity.rel_pure_yaw_envs=0.15
    env.commands.base_velocity.rel_pure_lateral_envs=0.15
    env.events.randomize_base_payload.params.payload_mass_range="[0.0,12.0]"
    env.events.randomize_base_payload.params.payload_zero_probability=0.35
    env.events.randomize_base_payload.params.payload_position_range.x="[-0.15,0.15]"
    env.events.randomize_base_payload.params.payload_position_range.y="[-0.10,0.10]"
    env.events.randomize_base_payload.params.payload_position_range.z="[0.05,0.20]"
    env.events.randomize_base_payload.params.base_mass_scale_range="[0.95,1.05]"
    env.events.randomize_base_payload.params.base_com_bias_range.x="[-0.02,0.02]"
    env.events.randomize_base_payload.params.base_com_bias_range.y="[-0.02,0.02]"
    env.events.randomize_base_payload.params.base_com_bias_range.z="[-0.02,0.02]"
    env.events.randomize_rigid_body_mass_others.params.mass_distribution_params="[0.7,1.3]"
    env.events.randomize_rigid_body_material.params.static_friction_range="[0.4,1.6]"
    env.events.randomize_rigid_body_material.params.dynamic_friction_range="[0.3,1.4]"
    env.events.randomize_rigid_body_material.params.restitution_range="[0.0,0.2]"
    env.events.randomize_reset_hips.params.position_range="[0.9,1.1]"
    env.events.randomize_actuator_gains.params.stiffness_distribution_params="[0.8,1.2]"
    env.events.randomize_actuator_gains.params.damping_distribution_params="[0.8,1.2]"
    env.events.randomize_apply_external_force_torque.params.force_range="[-30.0,30.0]"
    env.events.randomize_apply_external_force_torque.params.torque_range="[-10.0,10.0]"
    env.events.randomize_push_robot.params.velocity_range.x="[-0.5,0.5]"
    env.events.randomize_push_robot.params.velocity_range.y="[-0.5,0.5]"
    env.events.randomize_reset_base.params.pose_range.roll="[-3.14,3.14]"
    env.events.randomize_reset_base.params.pose_range.pitch="[-3.14,3.14]"
    env.events.randomize_reset_base.params.pose_range.yaw="[-3.14,3.14]"
    env.events.randomize_reset_base.params.velocity_range.x="[-0.5,0.5]"
    env.events.randomize_reset_base.params.velocity_range.y="[-0.5,0.5]"
    env.events.randomize_reset_base.params.velocity_range.z="[-0.5,0.5]"
    env.events.randomize_reset_base.params.velocity_range.roll="[-0.5,0.5]"
    env.events.randomize_reset_base.params.velocity_range.pitch="[-0.5,0.5]"
    env.events.randomize_reset_base.params.velocity_range.yaw="[-0.5,0.5]"
    env.rewards.joint_torques_l2.weight=-3.0e-5
    env.rewards.joint_power.weight=-1.0e-5
    env.rewards.wheel_torques_l2.weight=-1.5e-3
    env.rewards.wheel_power.weight=-3.0e-5
    env.rewards.track_lin_vel_xy_exp.weight=3.75
    env.rewards.track_ang_vel_z_exp.weight=1.875
    env.rewards.flat_orientation_l2.weight=-1.0
    env.rewards.hip_abs_limit_l2.weight=-5.0
    env.rewards.hip_abs_limit_l2.params.limit=0.28
    env.rewards.action_rate_l2.weight=-0.015
    env.rewards.joint_acc_l2.weight=-3.0e-7
    env.rewards.joint_acc_wheel_l2.weight=-5.0e-10
    env.rewards.wheel_contact_slip_l2.weight=-0.05
    env.rewards.mode_swing_clearance_l2.weight=-5.0
    env.rewards.minimum_wheel_contacts_l2.weight=-0.5
    env.rewards.touchdown_velocity_l2.weight=-0.2
    env.rewards.contact_forces.params.threshold=450.0
)

manifest() {
    local stage="$1" iterations="$2" source="$3" checkpoint="$4"; shift 4
    "$PYTHON" - "$OUTPUT/manifests/stages.ndjson" "$stage" "$iterations" "$source" "$checkpoint" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, stage, iterations, source, checkpoint, *arguments = sys.argv[1:]
with open(path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "iterations": int(iterations),
        "source": source,
        "checkpoint": checkpoint,
        "arguments": arguments,
    }, sort_keys=True) + "\n")
PY
}

latency_arguments() {
    local argument index values=("$@")
    LATENCY_ARGS=()
    for ((index = 0; index < ${#values[@]}; index++)); do
        argument="${values[index]}"
        case "$argument" in
            --obs-delay-min-frames|--obs-delay-max-frames|--obs-drop-start-prob|--obs-drop-max-frames|--action-delay-min-frames|--action-delay-max-frames)
                LATENCY_ARGS+=("$argument" "${values[index + 1]}")
                ((index += 1))
                ;;
        esac
    done
}

smoke_checkpoint() {
    local stage="$1" checkpoint="$2"; shift 2
    local output="$OUTPUT/results/${stage}_smoke32.json"
    latency_arguments "$@"
    record "SMOKE32_START stage=$stage checkpoint=$checkpoint"
    "$PYTHON" "$EVALUATE" --checkpoint "$checkpoint" --output "$output" \
        --kp 90.0 --kd 6.93 --seed 20260901 --num-envs-per-scenario 8 \
        --warmup-steps 0 --sample-steps 32 --recovery-hold-steps 8 \
        --payload-mass 0 --payload-profile center --device cuda:0 --scenarios stand \
        "${LATENCY_ARGS[@]}" 2>&1 | tee "$OUTPUT/logs/smoke_${stage}.log"
    "$PYTHON" "$GATE" smoke "$output" --output "$OUTPUT/results/${stage}_smoke32_gate.json" || {
        record "SMOKE32_FAILED stage=$stage"
        return 1
    }
    record "SMOKE32_PASS stage=$stage"
}

evaluate_checkpoints() {
    local stage="$1" run_dir="$2"; shift 2
    local checkpoint final_checkpoint stem number payload label output
    local latency_args=("$@")
    final_checkpoint="$(latest_checkpoint "$run_dir")"
    while IFS= read -r checkpoint; do
        stem="${checkpoint##*/model_}"; number="${stem%.pt}"
        if [[ "$checkpoint" != "$final_checkpoint" ]] && (( number % 500 != 0 && (number + 1) % 500 != 0 )); then
            continue
        fi
        local envs=256 warmup=200 samples=600 suffix="" scenarios=(
            stand forward_0.6 backward_-0.6 left_0.6 right_-0.6 yaw_left_0.6 yaw_right_-0.6
        ) payloads=(0 6 12)
        if [[ "$checkpoint" != "$final_checkpoint" ]]; then
            envs=64; warmup=100; samples=200; suffix="_screen"
            scenarios=(stand forward_0.6 left_0.6 yaw_left_0.6)
            payloads=(0 12)
        fi
        for payload in "${payloads[@]}"; do
            for label in zero trained; do
                [[ "$label" == trained && ${#latency_args[@]} -eq 0 ]] && continue
                output="$OUTPUT/results/${stage}_${number}${suffix}_payload${payload}_${label}.json"
                [[ -s "$output" ]] && continue
                local active_args=()
                [[ "$label" == trained ]] && active_args=("${latency_args[@]}")
                "$PYTHON" "$EVALUATE" --checkpoint "$checkpoint" --output "$output" \
                    --kp 90.0 --kd 6.93 --seed 20260901 --num-envs-per-scenario "$envs" \
                    --warmup-steps "$warmup" --sample-steps "$samples" --payload-mass "$payload" \
                    --payload-profile center --device cuda:0 \
                    --scenarios "${scenarios[@]}" \
                    "${active_args[@]}" 2>&1 | tee "$OUTPUT/logs/eval_${stage}_${number}_payload${payload}_${label}.log"
            done
        done
    done < <(find "$run_dir" -maxdepth 1 -type f -name 'model_*.pt' -print | sort -V)
}

evaluate_full_checkpoint() {
    local checkpoint="$1" output="$2"
    "$PYTHON" "$EVALUATE" --checkpoint "$checkpoint" --output "$output" \
        --kp 90.0 --kd 6.93 --seed 20260901 --num-envs-per-scenario 256 \
        --warmup-steps 200 --sample-steps 600 --payload-mass 0 \
        --payload-profile center --device cuda:0 \
        --scenarios stand forward_0.6 backward_-0.6 left_0.6 right_-0.6 yaw_left_0.6 yaw_right_-0.6
}

evaluate_command_profiles() {
    local checkpoint="$1" stage="$2"; shift 2
    local profile output
    for profile in step ramp; do
        output="$OUTPUT/results/${stage}_${profile}.json"
        [[ -s "$output" ]] && continue
        record "COMMAND_PROFILE_EVAL stage=$stage profile=$profile checkpoint=$checkpoint"
        "$PYTHON" "$EVALUATE" --checkpoint "$checkpoint" --output "$output" \
            --kp 90.0 --kd 6.93 --seed 20260901 --num-envs-per-scenario 64 \
            --warmup-steps 100 --sample-steps 400 --command-profile "$profile" \
            --command-transition-steps 50 --payload-mass 0 --payload-profile center \
            --device cuda:0 \
            --scenarios stand forward_0.6 backward_-0.6 left_0.6 right_-0.6 yaw_left_0.6 yaw_right_-0.6 \
            "$@" 2>&1 | tee "$OUTPUT/logs/command_profile_${stage}_${profile}.log"
    done
}

evaluate_latency_matrix() {
    local stage="$1" checkpoint="$2" frames kind output source_zero
    local scenarios=(stand forward_0.6 backward_-0.6 left_0.6 right_-0.6 yaw_left_0.6 yaw_right_-0.6)
    for frames in 0 1 2 3; do
        for kind in obs drop act combined; do
            output="$OUTPUT/results/${stage}_matrix_${kind}_${frames}f.json"
            [[ -s "$output" ]] && continue
            if (( frames == 0 )) && [[ "$kind" != obs ]]; then
                source_zero="$OUTPUT/results/${stage}_matrix_obs_0f.json"
                [[ -s "$source_zero" ]] || { record "MATRIX_ZERO_REQUIRED $source_zero"; return 1; }
                cp "$source_zero" "$output"
                continue
            fi
            local args=()
            case "$kind" in
                obs) args=(--obs-delay-min-frames "$frames" --obs-delay-max-frames "$frames") ;;
                drop)
                    args=(--obs-drop-start-prob 0.02 --obs-drop-max-frames "$frames")
                    ;;
                act) args=(--action-delay-min-frames "$frames" --action-delay-max-frames "$frames") ;;
                combined)
                    args=(--obs-delay-min-frames "$frames" --obs-delay-max-frames "$frames"
                          --obs-drop-start-prob 0.02 --obs-drop-max-frames "$frames"
                          --action-delay-min-frames "$frames" --action-delay-max-frames "$frames")
                    ;;
            esac
            record "MATRIX_EVAL stage=$stage kind=$kind frames=$frames checkpoint=$checkpoint"
            "$PYTHON" "$EVALUATE" --checkpoint "$checkpoint" --output "$output" \
                --kp 90.0 --kd 6.93 --seed 20260901 --num-envs-per-scenario 256 \
                --warmup-steps 200 --sample-steps 600 --payload-mass 0 \
                --payload-profile center --device cuda:0 --scenarios "${scenarios[@]}" \
                "${args[@]}" 2>&1 | tee "$OUTPUT/logs/matrix_${stage}_${kind}_${frames}f.log"
        done
    done
}

record_gate() {
    local label="$1" mode="$2" output="$3"; shift 3
    if "$PYTHON" "$GATE" "$mode" "$@" --output "$output"; then
        record "HARD_GATE_PASS label=$label result=$output"
    else
        record "HARD_GATE_FAIL label=$label result=$output"
        return 1
    fi
}

finish_stage() {
    local stage="$1" iterations="$2" source="$3"; shift 3
    local all_args=("$@") evaluation_args=() number
    latency_arguments "${all_args[@]}"; evaluation_args=("${LATENCY_ARGS[@]}")
    LAST_RUN="$(latest_run "$stage")"
    [[ -n "$LAST_RUN" ]] || { record "RUN_NOT_FOUND stage=$stage"; return 1; }
    LAST_CHECKPOINT="$(latest_checkpoint "$LAST_RUN")"
    [[ -f "$LAST_CHECKPOINT" ]] || { record "CHECKPOINT_NOT_FOUND stage=$stage"; return 1; }
    manifest "$stage" "$iterations" "$source" "$LAST_CHECKPOINT" "${all_args[@]}"
    evaluate_checkpoints "$stage" "$LAST_RUN" "${evaluation_args[@]}"
    number="${LAST_CHECKPOINT##*/model_}"; number="${number%.pt}"
    if [[ "$stage" == g1r_* ]]; then
        record_gate "$stage" nominal "$OUTPUT/results/${stage}_${number}_nominal_gate.json" \
            "$OUTPUT/results/${stage}_${number}_payload0_zero.json" || true
    elif [[ "$stage" == lat_* ]]; then
        [[ -s "$OUTPUT/results/nominal_winner_eval.json" ]] || {
            record "NOMINAL_WINNER_EVAL_REQUIRED stage=$stage"
            return 1
        }
        if [[ "$stage" == lat_combined_3f ]]; then
            record_gate "$stage" latency "$OUTPUT/results/${stage}_${number}_latency_gate.json" \
                "$OUTPUT/results/${stage}_${number}_payload0_zero.json" \
                "$OUTPUT/results/${stage}_${number}_payload0_trained.json" \
                --nominal-result "$OUTPUT/results/nominal_winner_eval.json"
        else
            record_gate "$stage" latency "$OUTPUT/results/${stage}_${number}_latency_gate.json" \
                "$OUTPUT/results/${stage}_${number}_payload0_zero.json" \
                "$OUTPUT/results/${stage}_${number}_payload0_trained.json" \
                --nominal-result "$OUTPUT/results/nominal_winner_eval.json" || true
        fi
    fi
    if [[ "$stage" == lat_combined_3f ]]; then
        evaluate_latency_matrix "$stage" "$LAST_CHECKPOINT"
        evaluate_command_profiles "$LAST_CHECKPOINT" "$stage" \
            --obs-delay-min-frames 3 --obs-delay-max-frames 3 \
            --obs-drop-start-prob 0.02 --obs-drop-max-frames 3 \
            --action-delay-min-frames 3 --action-delay-max-frames 3
        record_gate "$stage-matrix" latency "$OUTPUT/results/${stage}_matrix_gate.json" \
            "$OUTPUT/results/${stage}_matrix_combined_0f.json" \
            "$OUTPUT/results/${stage}_matrix_combined_3f.json" \
            --nominal-result "$OUTPUT/results/nominal_winner_eval.json"
    fi
    record "COMPLETE $stage checkpoint=$LAST_CHECKPOINT"
}

train_actor_stage() {
    local stage="$1" iterations="$2" source="$3"; shift 3
    local extra=("$@")
    smoke_checkpoint "$stage" "$source" "${extra[@]}"
    record "START $stage source=$source actor_warm_start=1 envs=4096 seed=42 iterations=$iterations"
    "$PYTHON" "$TRAIN" "${COMMON[@]}" --max_iterations "$iterations" --run_name "$stage" \
        --actor-warm-start "$source" "${extra[@]}" \
        2>&1 | tee "$OUTPUT/logs/train_${stage}.log"
    finish_stage "$stage" "$iterations" "$source" "${extra[@]}"
}

train_resume_stage() {
    local stage="$1" iterations="$2" source_run="$3" source_checkpoint="$4"; shift 4
    local extra=("$@")
    smoke_checkpoint "$stage" "$source_checkpoint" "${extra[@]}"
    record "START $stage source=$source_checkpoint full_resume=1 envs=4096 seed=42 iterations=$iterations"
    "$PYTHON" "$TRAIN" "${COMMON[@]}" --max_iterations "$iterations" --run_name "$stage" \
        --resume --load_run "$(basename "$source_run")" --checkpoint "$(basename "$source_checkpoint")" \
        agent.algorithm.learning_rate=2.0e-4 agent.algorithm.schedule=fixed "${extra[@]}" \
        2>&1 | tee "$OUTPUT/logs/train_${stage}.log"
    finish_stage "$stage" "$iterations" "$source_checkpoint" "${extra[@]}"
}

run_nominal_branch() {
    local stage="$1" iterations="$2" mirror_weight="$3"; shift 3
    train_actor_stage "$stage" "$iterations" "$G1_SOURCE" \
        env.rewards.joint_mirror.weight="$mirror_weight" "$@"
}

run_nominal() {
    [[ -f "$G1_SOURCE" ]] || { record "G1_SOURCE_NOT_FOUND $G1_SOURCE"; exit 2; }
    run_nominal_branch g1r_a 2000 -0.10
    run_nominal_branch g1r_b 2000 -0.15 \
        env.rewards.joint_mirror.params.command_name=base_velocity \
        env.rewards.joint_mirror.params.standing_scale=1.0 \
        env.rewards.joint_mirror.params.pure_yaw_scale=0.5 \
        env.rewards.joint_mirror.params.pure_lateral_scale=0.0 \
        env.rewards.joint_mirror.params.mixed_scale=1.0
    record "NOMINAL_COMPLETE write selected checkpoint to $NOMINAL_WINNER_FILE"
}

run_nominal_recover() {
    local run_a checkpoint_a checkpoint_number remaining
    run_a="$(latest_run g1r_a)"
    [[ -n "$run_a" ]] || { record "G1R_A_RUN_REQUIRED"; exit 2; }
    checkpoint_a="$(latest_checkpoint "$run_a")"
    checkpoint_number="${checkpoint_a##*/model_}"; checkpoint_number="${checkpoint_number%.pt}"
    remaining=$((1999 - checkpoint_number))
    (( remaining > 0 )) || { record "G1R_A_ALREADY_COMPLETE checkpoint=$checkpoint_a"; exit 2; }
    record "RECOVER g1r_a checkpoint=$checkpoint_a remaining_iterations=$remaining"
    train_resume_stage g1r_a_recover_2k "$remaining" "$run_a" "$checkpoint_a" \
        env.rewards.joint_mirror.weight=-0.10
    run_nominal_branch g1r_b 2000 -0.15 \
        env.rewards.joint_mirror.params.command_name=base_velocity \
        env.rewards.joint_mirror.params.standing_scale=1.0 \
        env.rewards.joint_mirror.params.pure_yaw_scale=0.5 \
        env.rewards.joint_mirror.params.pure_lateral_scale=0.0 \
        env.rewards.joint_mirror.params.mixed_scale=1.0
    record "NOMINAL_COMPLETE recovered_from=$checkpoint_a write selected checkpoint to $NOMINAL_WINNER_FILE"
}

run_nominal_b() {
    local run_b checkpoint_b checkpoint_number remaining
    run_b="$(latest_run_any '*_g1r_b' '*_g1r_b_recover_2k')"
    checkpoint_b=""
    [[ -z "$run_b" ]] || checkpoint_b="$(latest_checkpoint "$run_b")"
    if [[ -n "$checkpoint_b" ]]; then
        checkpoint_number="${checkpoint_b##*/model_}"; checkpoint_number="${checkpoint_number%.pt}"
        remaining=$((1999 - checkpoint_number))
        if (( remaining <= 0 )); then
            record "G1R_B_ALREADY_COMPLETE checkpoint=$checkpoint_b"
            return
        fi
        train_resume_stage g1r_b_recover_2k "$remaining" "$run_b" "$checkpoint_b" \
            env.rewards.joint_mirror.weight=-0.15 \
            env.rewards.joint_mirror.params.command_name=base_velocity \
            env.rewards.joint_mirror.params.standing_scale=1.0 \
            env.rewards.joint_mirror.params.pure_yaw_scale=0.5 \
            env.rewards.joint_mirror.params.pure_lateral_scale=0.0 \
            env.rewards.joint_mirror.params.mixed_scale=1.0
    else
        run_nominal_branch g1r_b 2000 -0.15 \
            env.rewards.joint_mirror.params.command_name=base_velocity \
            env.rewards.joint_mirror.params.standing_scale=1.0 \
            env.rewards.joint_mirror.params.pure_yaw_scale=0.5 \
            env.rewards.joint_mirror.params.pure_lateral_scale=0.0 \
            env.rewards.joint_mirror.params.mixed_scale=1.0
    fi
}

run_nominal_extend() {
    local run_a run_b checkpoint_a checkpoint_b
    run_a="$(latest_run_any '*_g1r_a' '*_g1r_a_recover_2k')"
    run_b="$(latest_run_any '*_g1r_b' '*_g1r_b_recover_2k')"
    [[ -n "$run_a" && -n "$run_b" ]] || {
        record "NOMINAL_BASE_RUNS_REQUIRED complete g1r_a and g1r_b first"
        exit 2
    }
    checkpoint_a="$(latest_checkpoint "$run_a")"
    checkpoint_b="$(latest_checkpoint "$run_b")"
    train_resume_stage g1r_a_extend_5k 3000 "$run_a" "$checkpoint_a" \
        env.rewards.joint_mirror.weight=-0.10
    train_resume_stage g1r_b_extend_5k 3000 "$run_b" "$checkpoint_b" \
        env.rewards.joint_mirror.weight=-0.15 \
        env.rewards.joint_mirror.params.command_name=base_velocity \
        env.rewards.joint_mirror.params.standing_scale=1.0 \
        env.rewards.joint_mirror.params.pure_yaw_scale=0.5 \
        env.rewards.joint_mirror.params.pure_lateral_scale=0.0 \
        env.rewards.joint_mirror.params.mixed_scale=1.0
    record "NOMINAL_EXTEND_COMPLETE total_iterations_per_branch=5000"
}

run_symmetric_recovery() {
    local run_a checkpoint_a checkpoint_number
    run_a="$(latest_run g1r_a_extend_5k)"
    [[ -n "$run_a" ]] || { record "G1R_A_5K_REQUIRED"; exit 2; }
    checkpoint_a="$(latest_checkpoint "$run_a")"
    checkpoint_number="${checkpoint_a##*/model_}"; checkpoint_number="${checkpoint_number%.pt}"
    (( checkpoint_number >= 4997 )) || {
        record "G1R_A_5K_INCOMPLETE checkpoint=$checkpoint_a"
        exit 2
    }
    train_resume_stage g1r_symmetric_recovery_8k 3000 "$run_a" "$checkpoint_a" \
        env.rewards.joint_mirror.weight=-0.15 \
        env.rewards.joint_mirror.params.command_name=base_velocity \
        env.rewards.joint_mirror.params.standing_scale=1.0 \
        env.rewards.joint_mirror.params.pure_yaw_scale=1.0 \
        env.rewards.joint_mirror.params.pure_lateral_scale=1.0 \
        env.rewards.joint_mirror.params.mixed_scale=1.0 \
        env.rewards.flat_orientation_l2.weight=-1.5 \
        env.rewards.standing_lin_vel_xy_l2.weight=-4.0 \
        env.rewards.standing_ang_vel_z_l2.weight=-1.0 \
        env.rewards.standing_wheel_vel_l2.weight=-0.02 \
        env.rewards.track_ang_vel_z_exp.weight=2.25 \
        env.rewards.joint_torques_l2.weight=-4.0e-5
}

run_yaw_specialization() {
    train_actor_stage yaw_specialization_g2_3k 3000 "$YAW_SOURCE" \
        env.commands.base_velocity.rel_standing_envs=0.10 \
        env.commands.base_velocity.rel_pure_yaw_envs=0.50 \
        env.commands.base_velocity.rel_pure_lateral_envs=0.05 \
        env.rewards.joint_mirror.weight=-0.20 \
        env.rewards.joint_mirror.params.command_name=base_velocity \
        env.rewards.joint_mirror.params.standing_scale=1.0 \
        env.rewards.joint_mirror.params.pure_yaw_scale=1.0 \
        env.rewards.joint_mirror.params.pure_lateral_scale=1.0 \
        env.rewards.joint_mirror.params.mixed_scale=1.0 \
        env.rewards.pure_yaw_wheel_mirror_l2.weight=-0.002 \
        env.rewards.track_ang_vel_z_exp.weight=2.5 \
        env.rewards.track_ang_vel_z_exp.params.std=0.25 \
        env.rewards.pure_yaw_lin_vel_xy_l2.weight=-3.0 \
        env.rewards.flat_orientation_l2.weight=-2.0 \
        env.rewards.ang_vel_xy_l2.weight=-0.10 \
        env.rewards.mode_swing_clearance_l2.weight=-8.0 \
        env.rewards.mode_swing_clearance_l2.params.yaw_clearance=0.04 \
        env.rewards.touchdown_velocity_l2.weight=-0.3
}

run_yaw_soft_touchdown() {
    local source_run source_checkpoint
    source_run="$(latest_run yaw_specialization_g2_3k)"
    source_checkpoint="${SOFT_TOUCHDOWN_SOURCE:-$(latest_checkpoint "$source_run")}"
    train_actor_stage yaw_soft_touchdown_2k 2000 "$source_checkpoint" \
        env.commands.base_velocity.rel_standing_envs=0.10 \
        env.commands.base_velocity.rel_pure_yaw_envs=0.50 \
        env.commands.base_velocity.rel_pure_lateral_envs=0.05 \
        env.rewards.joint_mirror.weight=-0.20 \
        env.rewards.joint_mirror.params.standing_scale=1.0 \
        env.rewards.joint_mirror.params.pure_yaw_scale=1.0 \
        env.rewards.joint_mirror.params.pure_lateral_scale=1.0 \
        env.rewards.joint_mirror.params.mixed_scale=1.0 \
        env.rewards.pure_yaw_wheel_mirror_l2.weight=-0.002 \
        env.rewards.track_ang_vel_z_exp.weight=2.5 \
        env.rewards.track_ang_vel_z_exp.params.std=0.25 \
        env.rewards.pure_yaw_lin_vel_xy_l2.weight=-3.0 \
        env.rewards.flat_orientation_l2.weight=-2.0 \
        env.rewards.ang_vel_xy_l2.weight=-0.10 \
        env.rewards.mode_swing_clearance_l2.weight=-8.0 \
        env.rewards.mode_swing_clearance_l2.params.yaw_clearance=0.04 \
        env.rewards.touchdown_velocity_l2.weight=-0.8 \
        env.rewards.touchdown_velocity_l2.params.velocity_threshold=0.10 \
        env.rewards.contact_forces.weight=-1.0e-2 \
        env.rewards.contact_forces.params.threshold=300.0
}

run_nominal_select() {
    local run_a run_b checkpoint_a checkpoint_b result_a result_b
    run_a="$(latest_run g1r_a_extend_5k)"; run_b="$(latest_run g1r_b_extend_5k)"
    [[ -n "$run_a" && -n "$run_b" ]] || {
        record "NOMINAL_5K_RUNS_REQUIRED complete both extension branches before selection"
        exit 2
    }
    checkpoint_a="$(latest_checkpoint "$run_a")"; checkpoint_b="$(latest_checkpoint "$run_b")"
    result_a="$OUTPUT/results/nominal_select_a.json"
    result_b="$OUTPUT/results/nominal_select_b.json"
    evaluate_full_checkpoint "$checkpoint_a" "$result_a" 2>&1 | tee "$OUTPUT/logs/nominal_select_a.log"
    evaluate_full_checkpoint "$checkpoint_b" "$result_b" 2>&1 | tee "$OUTPUT/logs/nominal_select_b.log"
    if ! "$PYTHON" "$GATE" select "$result_a" "$result_b" \
        --checkpoints "$checkpoint_a" "$checkpoint_b" --winner-file "$NOMINAL_WINNER_FILE" \
        --output "$OUTPUT/results/nominal_selection.json"; then
        record "NOMINAL_SELECTION_FAILED no candidate passed every nominal hard gate; latency stages blocked"
        exit 1
    fi
    if [[ "$(<"$NOMINAL_WINNER_FILE")" == "$checkpoint_a" ]]; then
        cp "$result_a" "$OUTPUT/results/nominal_winner_eval.json"
    else
        cp "$result_b" "$OUTPUT/results/nominal_winner_eval.json"
    fi
    evaluate_command_profiles "$(<"$NOMINAL_WINNER_FILE")" nominal_winner
    record "NOMINAL_SELECTION_PASS winner=$(<"$NOMINAL_WINNER_FILE")"
}

resolve_nominal_winner() {
    if [[ -z "$NOMINAL_WINNER" && -s "$NOMINAL_WINNER_FILE" ]]; then
        IFS= read -r NOMINAL_WINNER < "$NOMINAL_WINNER_FILE"
    fi
    [[ -f "$NOMINAL_WINNER" ]] || {
        record "NOMINAL_WINNER_REQUIRED set NOMINAL_WINNER=/path/model.pt or write it to $NOMINAL_WINNER_FILE"
        exit 2
    }
}

run_orientation_recovery() {
    resolve_nominal_winner
    train_actor_stage g1r_orientation_recovery 1000 "$NOMINAL_WINNER" \
        env.rewards.flat_orientation_l2.weight=-1.5
}

run_latency_probes() {
    resolve_nominal_winner
    train_actor_stage lat_obs 500 "$NOMINAL_WINNER" \
        --obs-delay-min-frames 0 --obs-delay-max-frames 3
    train_actor_stage lat_drop 500 "$NOMINAL_WINNER" \
        --obs-drop-start-prob 0.02 --obs-drop-max-frames 3
    train_actor_stage lat_act 500 "$NOMINAL_WINNER" \
        --action-delay-min-frames 0 --action-delay-max-frames 3
}

run_latency_combined() {
    resolve_nominal_winner
    train_actor_stage lat_combined_1f 1000 "$NOMINAL_WINNER" \
        --obs-delay-min-frames 0 --obs-delay-max-frames 1 \
        --obs-drop-start-prob 0.02 --obs-drop-max-frames 1 \
        --action-delay-min-frames 0 --action-delay-max-frames 1
    local previous_run="$LAST_RUN" previous_checkpoint="$LAST_CHECKPOINT"
    train_resume_stage lat_combined_2f 1000 "$previous_run" "$previous_checkpoint" \
        --obs-delay-min-frames 0 --obs-delay-max-frames 2 \
        --obs-drop-start-prob 0.02 --obs-drop-max-frames 2 \
        --action-delay-min-frames 0 --action-delay-max-frames 2
    previous_run="$LAST_RUN"; previous_checkpoint="$LAST_CHECKPOINT"
    train_resume_stage lat_combined_3f 2000 "$previous_run" "$previous_checkpoint" \
        --obs-delay-min-frames 0 --obs-delay-max-frames 3 \
        --obs-drop-start-prob 0.02 --obs-drop-max-frames 3 \
        --action-delay-min-frames 0 --action-delay-max-frames 3
}

trap 'record "G1R_FAILED mode=$MODE line=$LINENO exit=$?"' ERR
case "$MODE" in
    yaw-specialization) run_yaw_specialization ;;
    yaw-soft-touchdown) run_yaw_soft_touchdown ;;
    symmetric-recovery) run_symmetric_recovery ;;
    orientation-recovery) run_orientation_recovery ;;
    latency-probes) run_latency_probes ;;
    latency-combined) run_latency_combined ;;
esac
record "G1R_MODE_COMPLETE mode=$MODE"
