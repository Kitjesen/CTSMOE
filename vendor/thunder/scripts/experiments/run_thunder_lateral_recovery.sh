#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${DOSO_ROOT:-/root/autodl-tmp/workspace/doso-train}"
PYTHON="${ISAAC_PYTHON:-/root/autodl-tmp/conda/envs/isaaclab/bin/python}"
TASK="DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0"
TRAIN="$ROOT/scripts/reinforcement_learning/rsl_rl/train.py"
RUN_ROOT="$ROOT/logs/rsl_rl/doso_thunder_terrain_inward_flat"
BASE_MODEL="${BASE_MODEL:-$RUN_ROOT/2026-09-01_03-29-35_fs20k_kp90_tw3e5_track125/model_13500.pt}"
OUTPUT="${LATERAL_OUTPUT:-$ROOT/outputs/thunder_lateral_smooth_4096_20260901}"
RUN_PREFIX="${RUN_PREFIX:-lateral_smooth}"
STAGE_ONE_RUN="${RUN_PREFIX}_g1_v2"
STAGE_TWO_RUN="${RUN_PREFIX}_g2_v2"
mkdir -p "$OUTPUT/logs"
cd "$ROOT"
export ACCEPT_EULA=Y

record() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$OUTPUT/progress.log"; }
latest_run() { find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*_$1" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-; }
latest_checkpoint() { find "$1" -maxdepth 1 -type f -name 'model_*.pt' -printf '%p\n' | sort -V | tail -n 1; }

COMMON=(
    --task "$TASK" --num_envs 4096 --seed 42 --headless --device cuda:0
    env.scene.robot.actuators.hip.stiffness=90.0
    env.scene.robot.actuators.hip.damping=6.93
    env.scene.robot.actuators.thigh.stiffness=90.0
    env.scene.robot.actuators.thigh.damping=6.93
    env.scene.robot.actuators.calf.stiffness=90.0
    env.scene.robot.actuators.calf.damping=6.93
    env.commands.base_velocity.rel_standing_envs=0.10
    env.commands.base_velocity.rel_pure_yaw_envs=0.20
    env.commands.base_velocity.rel_pure_lateral_envs=0.30
    env.events.randomize_rigid_body_material.params.static_friction_range="[0.4,1.6]"
    env.events.randomize_rigid_body_material.params.dynamic_friction_range="[0.3,1.4]"
    env.events.randomize_rigid_body_material.params.restitution_range="[0.0,0.2]"
    env.events.randomize_reset_hips.params.position_range="[0.9,1.1]"
    env.events.randomize_actuator_gains.params.stiffness_distribution_params="[0.8,1.2]"
    env.events.randomize_actuator_gains.params.damping_distribution_params="[0.8,1.2]"
    env.events.randomize_reset_base.params.pose_range.z="[0.0,0.05]"
    env.events.randomize_reset_base.params.pose_range.roll="[-0.25,0.25]"
    env.events.randomize_reset_base.params.pose_range.pitch="[-0.25,0.25]"
    env.events.randomize_reset_base.params.velocity_range.x="[-0.2,0.2]"
    env.events.randomize_reset_base.params.velocity_range.y="[-0.2,0.2]"
    env.events.randomize_reset_base.params.velocity_range.z="[-0.2,0.2]"
    env.events.randomize_reset_base.params.velocity_range.roll="[-0.2,0.2]"
    env.events.randomize_reset_base.params.velocity_range.pitch="[-0.2,0.2]"
    env.events.randomize_reset_base.params.velocity_range.yaw="[-0.2,0.2]"
    env.rewards.track_lin_vel_xy_exp.weight=3.75
    env.rewards.track_ang_vel_z_exp.weight=1.875
    env.rewards.hip_abs_limit_l2.params.limit=0.28
    env.rewards.joint_mirror.weight=-0.10
)

train_stage_one() {
    local variant="$STAGE_ONE_RUN"
    record "START $variant actor_warm_start=$(basename "$BASE_MODEL") envs=4096 iterations=1000 eta_min=14"
    "$PYTHON" "$TRAIN" "${COMMON[@]}" --max_iterations 1000 --run_name "$variant" \
        --actor-warm-start "$BASE_MODEL" \
        env.events.randomize_base_payload.params.payload_mass_range="[0.0,2.0]" \
        env.events.randomize_base_payload.params.payload_zero_probability=0.70 \
        env.events.randomize_rigid_body_mass_others.params.mass_distribution_params="[0.9,1.1]" \
        env.events.randomize_apply_external_force_torque.params.force_range="[-5.0,5.0]" \
        env.events.randomize_apply_external_force_torque.params.torque_range="[-1.5,1.5]" \
        env.events.randomize_push_robot.params.velocity_range.x="[-0.2,0.2]" \
        env.events.randomize_push_robot.params.velocity_range.y="[-0.2,0.2]" \
        env.rewards.flat_orientation_l2.weight=-1.0 \
        env.rewards.hip_abs_limit_l2.weight=-5.0 \
        env.rewards.action_rate_l2.weight=-0.015 \
        env.rewards.joint_acc_l2.weight=-3.0e-7 \
        env.rewards.joint_acc_wheel_l2.weight=-5.0e-10 \
        env.rewards.wheel_contact_slip_l2.weight=-0.05 \
        env.rewards.mode_swing_clearance_l2.weight=-5.0 \
        env.rewards.minimum_wheel_contacts_l2.weight=-0.5 \
        2>&1 | tee "$OUTPUT/logs/train_${variant}.log"
}

train_stage_two() {
    local source_run="$1" source_checkpoint="$2" variant="$STAGE_TWO_RUN"
    record "START $variant resume=$(basename "$source_checkpoint") envs=4096 iterations=4000 eta_min=55"
    "$PYTHON" "$TRAIN" "${COMMON[@]}" --max_iterations 4000 --run_name "$variant" \
        --resume --load_run "$(basename "$source_run")" --checkpoint "$(basename "$source_checkpoint")" \
        agent.algorithm.learning_rate=2.0e-4 \
        agent.algorithm.schedule=fixed \
        env.events.randomize_base_payload.params.payload_mass_range="[0.0,4.0]" \
        env.events.randomize_base_payload.params.payload_zero_probability=0.50 \
        env.events.randomize_rigid_body_mass_others.params.mass_distribution_params="[0.85,1.15]" \
        env.events.randomize_apply_external_force_torque.params.force_range="[-10.0,10.0]" \
        env.events.randomize_apply_external_force_torque.params.torque_range="[-3.0,3.0]" \
        env.events.randomize_push_robot.params.velocity_range.x="[-0.3,0.3]" \
        env.events.randomize_push_robot.params.velocity_range.y="[-0.3,0.3]" \
        env.rewards.flat_orientation_l2.weight=-2.0 \
        env.rewards.hip_abs_limit_l2.weight=-20.0 \
        env.rewards.joint_mirror.weight=-0.15 \
        env.rewards.action_rate_l2.weight=-0.02 \
        env.rewards.joint_acc_l2.weight=-5.0e-7 \
        env.rewards.joint_acc_wheel_l2.weight=-1.0e-9 \
        env.rewards.wheel_contact_slip_l2.weight=-0.20 \
        env.rewards.mode_swing_clearance_l2.weight=-100.0 \
        env.rewards.minimum_wheel_contacts_l2.weight=-1.0 \
        2>&1 | tee "$OUTPUT/logs/train_${variant}.log"
}

[[ -f "$BASE_MODEL" ]] || { record "BASE_MODEL_NOT_FOUND $BASE_MODEL"; exit 2; }
trap 'record "LATERAL_SMOOTH_FAILED line=$LINENO exit=$?"' ERR
while screen -list 2>/dev/null | grep -q '[.]doso-post20k'; do
    record "WAIT doso-post20k"
    sleep 60
done

train_stage_one
stage_one_run="$(latest_run "$STAGE_ONE_RUN")"
stage_one_checkpoint="$(latest_checkpoint "$stage_one_run")"
record "COMPLETE $STAGE_ONE_RUN checkpoint=$stage_one_checkpoint"

train_stage_two "$stage_one_run" "$stage_one_checkpoint"
stage_two_run="$(latest_run "$STAGE_TWO_RUN")"
stage_two_checkpoint="$(latest_checkpoint "$stage_two_run")"
record "COMPLETE $STAGE_TWO_RUN checkpoint=$stage_two_checkpoint"
record "LATERAL_SMOOTH_4096_COMPLETE prefix=$RUN_PREFIX total_iterations=5000"
