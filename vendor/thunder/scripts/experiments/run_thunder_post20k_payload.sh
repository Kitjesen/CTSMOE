#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${DOSO_ROOT:-/root/autodl-tmp/workspace/doso-train}"
PYTHON="${ISAAC_PYTHON:-/root/autodl-tmp/conda/envs/isaaclab/bin/python}"
TASK="DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0"
TRAIN="$ROOT/scripts/reinforcement_learning/rsl_rl/train.py"
EVALUATE="$ROOT/scripts/reinforcement_learning/rsl_rl/evaluate_torque_tracking.py"
RUN_ROOT="$ROOT/logs/rsl_rl/doso_thunder_terrain_inward_flat"
OUTPUT="${POST20K_OUTPUT:-$ROOT/outputs/thunder_post20k_payload_20260901}"
MODEL_19999="${MODEL_19999:-}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/results" "$OUTPUT/manifests"
cd "$ROOT"
export ACCEPT_EULA=Y

record() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$OUTPUT/progress.log"; }
latest_checkpoint() { find "$1" -maxdepth 1 -type f -name 'model_*.pt' -printf '%p\n' | sort -V | tail -n 1; }
latest_run() { find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*_$1" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-; }

if [[ -z "$MODEL_19999" ]]; then
    MODEL_19999="$(find "$RUN_ROOT" -path '*fs20k_kp90_tw3e5_track125*/model_19999.pt' -print | sort | tail -n 1)"
fi
[[ -f "$MODEL_19999" ]] || { record "MODEL_19999_NOT_FOUND set MODEL_19999=/path/model_19999.pt"; exit 2; }

manifest() {
    "$PYTHON" - "$OUTPUT/manifests/stages.ndjson" "$@" <<'PY'
import json, sys
from datetime import datetime, timezone
path, stage, iterations, source, checkpoint = sys.argv[1:]
with open(path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "stage": stage,
        "iterations": int(iterations), "source": source, "checkpoint": checkpoint}) + "\n")
PY
}

screen_checkpoints() {
    local stage="$1" run_dir="$2" kp=90 kd=6.93 checkpoint stem number mass
    while IFS= read -r checkpoint; do
        stem="${checkpoint##*/model_}"; number="${stem%.pt}"
        if (( number % 500 != 0 && (number + 1) % 500 != 0 )); then continue; fi
        for mass in 0 12; do
            [[ -s "$OUTPUT/results/${stage}_${number}_payload${mass}_screen.json" ]] && continue
            "$PYTHON" "$EVALUATE" --checkpoint "$checkpoint" \
                --output "$OUTPUT/results/${stage}_${number}_payload${mass}_screen.json" \
                --kp "$kp" --kd "$kd" --seed 20260901 --num-envs-per-scenario 64 \
                --warmup-steps 100 --sample-steps 200 --payload-mass "$mass" --payload-profile center \
                --scenarios stand forward_0.6 left_0.6 yaw_left_0.6 --device cuda:0 \
                2>&1 | tee "$OUTPUT/logs/eval_${stage}_${number}_payload${mass}.log"
        done
    done < <(find "$run_dir" -maxdepth 1 -type f -name 'model_*.pt' -print | sort -V)
}

run_stage() {
    local stage="$1" iterations="$2" payload_max="$3" zero_prob="$4"
    local px="$5" py="$6" pz0="$7" pz1="$8" static0="$9" static1="${10}"
    local dynamic0="${11}" dynamic1="${12}" other0="${13}" other1="${14}"
    local force="${15}" torque="${16}" push="${17}" wheel_torque="${18}" wheel_power="${19}"
    local run_dir checkpoint source_args base0 base1 com reset_angle reset_velocity
    source_args=(--actor-warm-start "$MODEL_19999")
    if [[ -n "${PREVIOUS_RUN:-}" ]]; then
        source_args=(--resume --load_run "$(basename "$PREVIOUS_RUN")" --checkpoint "$(basename "$PREVIOUS_CHECKPOINT")")
    fi
    case "$stage" in
        s1_*) base0=1.0; base1=1.0; com=0.0; reset_angle=0.20; reset_velocity=0.10 ;;
        s2_*) base0=0.99; base1=1.01; com=0.005; reset_angle=0.50; reset_velocity=0.20 ;;
        s3_*) base0=0.97; base1=1.03; com=0.01; reset_angle=1.50; reset_velocity=0.35 ;;
        *) base0=0.95; base1=1.05; com=0.02; reset_angle=3.14; reset_velocity=0.50 ;;
    esac

    record "START $stage iterations=$iterations envs=32768 seed=42 payload=0-$payload_max zero_probability=$zero_prob"
    "$PYTHON" "$TRAIN" --task "$TASK" --num_envs 32768 --seed 42 --max_iterations "$iterations" \
        --run_name "$stage" --headless --device cuda:0 "${source_args[@]}" \
        env.scene.robot.actuators.hip.stiffness=90.0 env.scene.robot.actuators.hip.damping=6.93 \
        env.scene.robot.actuators.thigh.stiffness=90.0 env.scene.robot.actuators.thigh.damping=6.93 \
        env.scene.robot.actuators.calf.stiffness=90.0 env.scene.robot.actuators.calf.damping=6.93 \
        env.rewards.joint_torques_l2.weight=-3e-5 env.rewards.joint_power.weight=-1e-5 \
        env.rewards.wheel_torques_l2.weight="$wheel_torque" env.rewards.wheel_power.weight="$wheel_power" \
        env.rewards.track_lin_vel_xy_exp.weight=3.75 env.rewards.track_ang_vel_z_exp.weight=1.875 \
        env.commands.base_velocity.rel_standing_envs=0.15 env.commands.base_velocity.rel_pure_yaw_envs=0.15 \
        env.commands.base_velocity.rel_pure_lateral_envs=0.10 \
        env.events.randomize_base_payload.params.payload_mass_range="[0.0,$payload_max]" \
        env.events.randomize_base_payload.params.payload_zero_probability="$zero_prob" \
        env.events.randomize_base_payload.params.payload_position_range.x="[-$px,$px]" \
        env.events.randomize_base_payload.params.payload_position_range.y="[-$py,$py]" \
        env.events.randomize_base_payload.params.payload_position_range.z="[$pz0,$pz1]" \
        env.events.randomize_base_payload.params.base_mass_scale_range="[$base0,$base1]" \
        env.events.randomize_base_payload.params.base_com_bias_range.x="[-$com,$com]" \
        env.events.randomize_base_payload.params.base_com_bias_range.y="[-$com,$com]" \
        env.events.randomize_base_payload.params.base_com_bias_range.z="[-$com,$com]" \
        env.events.randomize_rigid_body_mass_others.params.mass_distribution_params="[$other0,$other1]" \
        env.events.randomize_rigid_body_material.params.static_friction_range="[$static0,$static1]" \
        env.events.randomize_rigid_body_material.params.dynamic_friction_range="[$dynamic0,$dynamic1]" \
        env.events.randomize_rigid_body_material.params.restitution_range="[0.0,0.2]" \
        env.events.randomize_actuator_gains.params.stiffness_distribution_params="[0.8,1.2]" \
        env.events.randomize_actuator_gains.params.damping_distribution_params="[0.8,1.2]" \
        env.events.randomize_apply_external_force_torque.params.force_range="[-$force,$force]" \
        env.events.randomize_apply_external_force_torque.params.torque_range="[-$torque,$torque]" \
        env.events.randomize_push_robot.params.velocity_range.x="[-$push,$push]" \
        env.events.randomize_push_robot.params.velocity_range.y="[-$push,$push]" \
        env.events.randomize_reset_base.params.pose_range.roll="[-$reset_angle,$reset_angle]" \
        env.events.randomize_reset_base.params.pose_range.pitch="[-$reset_angle,$reset_angle]" \
        env.events.randomize_reset_base.params.velocity_range.x="[-$reset_velocity,$reset_velocity]" \
        env.events.randomize_reset_base.params.velocity_range.y="[-$reset_velocity,$reset_velocity]" \
        env.events.randomize_reset_base.params.velocity_range.z="[-$reset_velocity,$reset_velocity]" \
        env.events.randomize_reset_base.params.velocity_range.roll="[-$reset_velocity,$reset_velocity]" \
        env.events.randomize_reset_base.params.velocity_range.pitch="[-$reset_velocity,$reset_velocity]" \
        env.events.randomize_reset_base.params.velocity_range.yaw="[-$reset_velocity,$reset_velocity]" \
        2>&1 | tee "$OUTPUT/logs/train_$stage.log"

    run_dir="$(latest_run "$stage")"; [[ -n "$run_dir" ]] || return 1
    checkpoint="$(latest_checkpoint "$run_dir")"; [[ -n "$checkpoint" ]] || return 1
    manifest "$stage" "$iterations" "${PREVIOUS_CHECKPOINT:-$MODEL_19999}" "$checkpoint"
    screen_checkpoints "$stage" "$run_dir"
    PREVIOUS_RUN="$run_dir"; PREVIOUS_CHECKPOINT="$checkpoint"
    record "COMPLETE $stage checkpoint=$checkpoint"
}

trap 'record "POST20K_FAILED line=$LINENO exit=$?"' ERR
run_stage s1_nominal_repair 1000 0.0 1.00 0.0 0.0 0.0 0.0 0.5 1.0 0.4 0.8 0.9 1.1 0.0 0.0 0.2 -7.5e-4 -1.5e-5
run_stage s2_payload_4kg 2000 4.0 0.50 0.05 0.04 0.05 0.10 0.4 1.05 0.3 0.85 0.85 1.15 10.0 3.0 0.3 -1.5e-3 -3e-5
run_stage s3_payload_8kg 2500 8.0 0.40 0.10 0.07 0.05 0.15 0.3 1.1 0.25 0.9 0.8 1.2 20.0 6.0 0.4 -1.5e-3 -3e-5
run_stage s4_full_payload 2500 12.0 0.35 0.15 0.10 0.05 0.20 0.25 1.2 0.2 1.0 0.7 1.3 30.0 10.0 0.5 -1.5e-3 -3e-5
record "POST20K_COMPLETE total_iterations=8000"
