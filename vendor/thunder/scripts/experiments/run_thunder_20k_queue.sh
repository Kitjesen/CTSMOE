#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp/workspace/doso-train"
PYTHON="/root/autodl-tmp/conda/envs/isaaclab/bin/python"
TASK="DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0"
TRAIN="$ROOT/scripts/reinforcement_learning/rsl_rl/train.py"
EVALUATE="$ROOT/scripts/reinforcement_learning/rsl_rl/evaluate_torque_tracking.py"
RUN_ROOT="$ROOT/logs/rsl_rl/doso_thunder_terrain_inward_flat"
OUTPUT="$ROOT/outputs/thunder_20k_full_dr_20260901"
PROGRESS="$OUTPUT/progress.log"

mkdir -p "$OUTPUT/logs" "$OUTPUT/results"
cd "$ROOT"
export ACCEPT_EULA=Y

record() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$PROGRESS"
}

latest_checkpoint() {
    find "$1" -maxdepth 1 -type f -name 'model_*.pt' -printf '%p\n' | sort -V | tail -n 1
}

run_one() {
    local name="$1" kp="$2" kd="$3" torque_weight="$4" tracking_scale="$5"
    local lin_weight yaw_weight run_dir checkpoint
    lin_weight="$($PYTHON -c "print(3.0 * float('$tracking_scale'))")"
    yaw_weight="$($PYTHON -c "print(1.5 * float('$tracking_scale'))")"

    if [[ -s "$OUTPUT/results/$name.json" ]]; then
        record "SKIP $name (result already exists)"
        return
    fi

    record "START $name from_scratch iterations=20000 envs=32768 seed=42 kp=$kp kd=$kd torque=$torque_weight tracking=${tracking_scale}x full_dr=on"
    "$PYTHON" "$TRAIN" \
        --task "$TASK" \
        --num_envs 32768 \
        --seed 42 \
        --max_iterations 20000 \
        --run_name "$name" \
        --headless \
        --device cuda:0 \
        env.scene.robot.init_state.pos=[0.0,0.0,0.4600] \
        env.scene.robot.init_state.joint_pos.fr_hip_joint=-0.20 \
        env.scene.robot.init_state.joint_pos.rl_hip_joint=-0.20 \
        env.scene.robot.init_state.joint_pos.fl_hip_joint=0.20 \
        env.scene.robot.init_state.joint_pos.rr_hip_joint=0.20 \
        env.rewards.joint_torques_l2.weight="$torque_weight" \
        env.rewards.track_lin_vel_xy_exp.weight="$lin_weight" \
        env.rewards.track_ang_vel_z_exp.weight="$yaw_weight" \
        env.scene.robot.actuators.hip.stiffness="$kp" \
        env.scene.robot.actuators.hip.damping="$kd" \
        env.scene.robot.actuators.thigh.stiffness="$kp" \
        env.scene.robot.actuators.thigh.damping="$kd" \
        env.scene.robot.actuators.calf.stiffness="$kp" \
        env.scene.robot.actuators.calf.damping="$kd" \
        2>&1 | tee "$OUTPUT/logs/train_$name.log"

    run_dir="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*_$name" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    [[ -n "$run_dir" ]] || { record "FAIL $name: run directory missing"; return 1; }
    checkpoint="$(latest_checkpoint "$run_dir")"
    [[ -n "$checkpoint" ]] || { record "FAIL $name: checkpoint missing"; return 1; }
    record "TRAINED $name checkpoint=$checkpoint"

    "$PYTHON" "$EVALUATE" \
        --checkpoint "$checkpoint" \
        --output "$OUTPUT/results/$name.json" \
        --pose 0.20 \
        --kp "$kp" \
        --kd "$kd" \
        --seed 20260901 \
        --num-envs-per-scenario 256 \
        --warmup-steps 200 \
        --sample-steps 600 \
        --device cuda:0 \
        2>&1 | tee "$OUTPUT/logs/evaluate_$name.log"
    record "EVALUATED $name result=$OUTPUT/results/$name.json"
}

trap 'record "QUEUE_FAILED line=$LINENO exit=$?"' ERR

# Highest-probability candidate first so the overnight run produces a useful model early.
run_one fs20k_kp90_tw3e5_track125 90.0 6.93 -3e-5 1.25
run_one fs20k_kp120_tw3e5_track125 120.0 8.00 -3e-5 1.25
run_one fs20k_kp60_tw3e5_track125 60.0 5.66 -3e-5 1.25
run_one fs20k_baseline_kp120_tw1e5_track100 120.0 8.00 -1e-5 1.00

record "QUEUE_COMPLETE"
