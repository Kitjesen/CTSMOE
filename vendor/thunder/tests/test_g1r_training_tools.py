import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/run_thunder_g1r_latency.sh"
DOC = ROOT / "docs/experiments/2026-09-02-thunder-g1r-latency.md"
GATE = ROOT / "scripts/experiments/g1r_result_gate.py"


def test_g1r_script_locks_nominal_training_contract():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "lateral_smooth_s4_7996_g1_v2/model_999.pt" in script
    allowed_modes = script.split('case "$MODE" in', 1)[1].split("*)", 1)[0]
    assert "symmetric-recovery" in allowed_modes
    assert "nominal-b" not in allowed_modes
    assert "nominal-extend" not in allowed_modes
    recovery = script.split("run_symmetric_recovery() {", 1)[1].split("\n}", 1)[0]
    for setting in (
        "g1r_symmetric_recovery_8k 3000",
        'checkpoint_number="${checkpoint_a##*/model_}"',
        "checkpoint_number >= 4997",
        "env.rewards.joint_mirror.weight=-0.15",
        "env.rewards.joint_mirror.params.standing_scale=1.0",
        "env.rewards.joint_mirror.params.pure_yaw_scale=1.0",
        "env.rewards.joint_mirror.params.pure_lateral_scale=1.0",
        "env.rewards.joint_mirror.params.mixed_scale=1.0",
        "env.rewards.flat_orientation_l2.weight=-1.5",
        "env.rewards.standing_lin_vel_xy_l2.weight=-4.0",
        "env.rewards.standing_ang_vel_z_l2.weight=-1.0",
        "env.rewards.standing_wheel_vel_l2.weight=-0.02",
        "env.rewards.track_ang_vel_z_exp.weight=2.25",
        "env.rewards.joint_torques_l2.weight=-4.0e-5",
    ):
        assert setting in recovery
    assert "--num_envs 4096 --seed 42" in script
    assert "--actor-warm-start" in script

    expected_overrides = {
        'rel_standing_envs=0.10',
        'rel_pure_yaw_envs=0.15',
        'rel_pure_lateral_envs=0.15',
        'payload_mass_range="[0.0,12.0]"',
        "payload_zero_probability=0.35",
        'mass_distribution_params="[0.7,1.3]"',
        'static_friction_range="[0.4,1.6]"',
        'dynamic_friction_range="[0.3,1.4]"',
        'force_range="[-30.0,30.0]"',
        'torque_range="[-10.0,10.0]"',
        'velocity_range.x="[-0.5,0.5]"',
        'velocity_range.y="[-0.5,0.5]"',
        'pose_range.roll="[-3.14,3.14]"',
        'pose_range.pitch="[-3.14,3.14]"',
        'pose_range.yaw="[-3.14,3.14]"',
        "flat_orientation_l2.weight=-1.0",
        "hip_abs_limit_l2.weight=-5.0",
        "hip_abs_limit_l2.params.limit=0.28",
        "action_rate_l2.weight=-0.015",
        "joint_acc_l2.weight=-3.0e-7",
        "joint_acc_wheel_l2.weight=-5.0e-10",
        "wheel_contact_slip_l2.weight=-0.05",
        "mode_swing_clearance_l2.weight=-5.0",
        "minimum_wheel_contacts_l2.weight=-0.5",
        "touchdown_velocity_l2.weight=-0.2",
        "contact_forces.params.threshold=450.0",
    }
    for override in expected_overrides:
        assert override in script

    assert "joint_mirror.params.command_name=base_velocity" in script
    assert "joint_mirror.params.standing_scale=1.0" in script
    assert "joint_mirror.params.mixed_scale=1.0" in script


def test_g1r_script_locks_yaw_specialization_contract():
    script = SCRIPT.read_text(encoding="utf-8")

    allowed_modes = script.split('case "$MODE" in', 1)[1].split("*)", 1)[0]
    assert "yaw-specialization" in allowed_modes
    assert "lateral_smooth_s4_7996_g2_v2/model_4998.pt" in script
    yaw = script.split("run_yaw_specialization() {", 1)[1].split("\n}", 1)[0]
    for setting in (
        "yaw_specialization_g2_3k 3000",
        "rel_standing_envs=0.10",
        "rel_pure_yaw_envs=0.50",
        "rel_pure_lateral_envs=0.05",
        "joint_mirror.weight=-0.20",
        "joint_mirror.params.pure_yaw_scale=1.0",
        "pure_yaw_wheel_mirror_l2.weight=-0.002",
        "track_ang_vel_z_exp.weight=2.5",
        "track_ang_vel_z_exp.params.std=0.25",
        "pure_yaw_lin_vel_xy_l2.weight=-3.0",
        "flat_orientation_l2.weight=-2.0",
        "ang_vel_xy_l2.weight=-0.10",
        "mode_swing_clearance_l2.weight=-8.0",
        "mode_swing_clearance_l2.params.yaw_clearance=0.04",
        "touchdown_velocity_l2.weight=-0.3",
    ):
        assert setting in yaw


def test_g1r_script_locks_soft_touchdown_contract():
    script = SCRIPT.read_text(encoding="utf-8")

    allowed_modes = script.split('case "$MODE" in', 1)[1].split("*)", 1)[0]
    assert "yaw-soft-touchdown" in allowed_modes
    stage = script.split("run_yaw_soft_touchdown() {", 1)[1].split("\n}", 1)[0]
    for setting in (
        "yaw_soft_touchdown_2k 2000",
        "latest_run yaw_specialization_g2_3k",
        "mode_swing_clearance_l2.weight=-8.0",
        "mode_swing_clearance_l2.params.yaw_clearance=0.04",
        "touchdown_velocity_l2.weight=-0.8",
        "touchdown_velocity_l2.params.velocity_threshold=0.10",
        "contact_forces.weight=-1.0e-2",
        "contact_forces.params.threshold=300.0",
    ):
        assert setting in stage


def test_g1r_script_encodes_latency_branches_and_combined_curriculum():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "lat_obs 500" in script
    assert "lat_drop 500" in script
    assert "lat_act 500" in script
    assert "lat_combined_1f 1000" in script
    assert "lat_combined_2f 1000" in script
    assert "lat_combined_3f 2000" in script
    assert "NOMINAL_WINNER" in script
    assert "--obs-delay-min-frames" in script
    assert "--obs-delay-max-frames" in script
    assert "--obs-drop-start-prob" in script
    assert "--obs-drop-max-frames" in script
    assert "--action-delay-min-frames" in script
    assert "--action-delay-max-frames" in script
    assert "payloads=(0 6 12)" in script
    assert "--num-envs-per-scenario 256" in script
    assert "--warmup-steps 200" in script
    assert "--sample-steps 600" in script
    assert "envs=64; warmup=100; samples=200" in script
    assert "payloads=(0 12)" in script
    assert "scenarios=(stand forward_0.6 left_0.6 yaw_left_0.6)" in script
    assert "stand forward_0.6 backward_-0.6 left_0.6 right_-0.6 yaw_left_0.6 yaw_right_-0.6" in script
    assert "smoke_checkpoint" in script
    assert "--sample-steps 32" in script
    assert "evaluate_latency_matrix" in script
    assert "for frames in 0 1 2 3" in script
    assert "obs drop act combined" in script
    assert "g1r_result_gate.py" in script
    assert "evaluate_command_profiles" in script
    assert "--command-profile \"$profile\"" in script
    assert "for profile in step ramp" in script
    record_gate = script.split("record_gate() {", 1)[1].split("\n}", 1)[0]
    assert "return 1" in record_gate
    matrix_gate = script.split('record_gate "$stage-matrix"', 1)[1].split("\n", 4)[:4]
    assert "|| true" not in "\n".join(matrix_gate)


def _result(tracking_rmse: float | None = None) -> dict:
    scenarios = {}
    targets = {
        "stand": (0.0, 0.0, 0.0),
        "forward_0.6": (0.6, 0.0, 0.0),
        "backward_-0.6": (-0.6, 0.0, 0.0),
        "left_0.6": (0.0, 0.6, 0.0),
        "right_-0.6": (0.0, -0.6, 0.0),
        "yaw_left_0.6": (0.0, 0.0, 0.6),
        "yaw_right_-0.6": (0.0, 0.0, -0.6),
    }
    for name, target in targets.items():
        g1_rmse = {
            "stand": 0.003372478,
            "forward_0.6": 0.004349761,
            "backward_-0.6": 0.012185449,
            "left_0.6": 0.036139850,
            "right_-0.6": 0.044075493,
            "yaw_left_0.6": 0.093986981,
            "yaw_right_-0.6": 0.043312497,
        }
        scenarios[name] = {
            "target_velocity": list(target),
            "actual_velocity_mean": list(target),
            "tracking_rmse": tracking_rmse if tracking_rmse is not None else g1_rmse[name] * 0.90,
            "base_attitude": {
                "roll_rms_rad": 0.001,
                "pitch_rms_rad": 0.001,
                "roll_peak_abs_rad": 0.005,
                "pitch_peak_abs_rad": 0.005,
            },
            "contact": {"history_peak_n": 250.0},
            "leg_torque": {"rms": 7.0, "over_90_limit_fraction": 0.0},
            "hip_torque": {"rms": 2.0},
            "wheel_torque": {"rms": 1.0, "over_90_limit_fraction": 0.0},
            "unexpected_done_fraction": 0.0,
            "finite": True,
        }
    return {"scenarios": scenarios, "passed": False}


def _run_gate(
    tmp_path: Path, mode: str, *results: dict, extra_args: tuple[str, ...] = ()
) -> tuple[subprocess.CompletedProcess[str], dict]:
    paths = []
    for index, result in enumerate(results):
        path = tmp_path / f"input_{index}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "gate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(GATE),
            mode,
            *(str(path) for path in paths),
            *extra_args,
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_g1r_gate_ignores_legacy_recovery_pass_and_enforces_nominal_hard_limits(tmp_path):
    result = _result()
    completed, gate = _run_gate(tmp_path, "nominal", result)
    assert completed.returncode == 0
    assert gate["passed"] is True

    result["scenarios"]["left_0.6"]["contact"]["history_peak_n"] = 450.0
    completed, gate = _run_gate(tmp_path, "nominal", result)
    assert completed.returncode == 1
    assert gate["checks"]["contact_peak_below_450n"] is False


def test_g1r_gate_uses_per_command_g1_limits_and_standing_torque_limits(tmp_path):
    result = _result()
    result["scenarios"]["forward_0.6"]["tracking_rmse"] = 0.004349761 * 1.06
    completed, gate = _run_gate(tmp_path, "nominal", result)
    assert completed.returncode == 1
    assert gate["checks"]["forward_0.6_rmse_at_most_g1_x1_05"] is False

    result = _result()
    result["scenarios"]["stand"]["leg_torque"]["rms"] = 7.486396799 + 0.001
    completed, gate = _run_gate(tmp_path, "nominal", result)
    assert completed.returncode == 1
    assert gate["checks"]["standing_leg_rms_at_most_g1"] is False


def test_g1r_gate_enforces_three_frame_combined_relative_limit(tmp_path):
    zero = _result(0.04)
    delayed = _result(0.047)
    nominal_path = tmp_path / "nominal.json"
    nominal_path.write_text(json.dumps(zero), encoding="utf-8")
    nominal_arg = ("--nominal-result", str(nominal_path))
    completed, gate = _run_gate(tmp_path, "latency", zero, delayed, extra_args=nominal_arg)
    assert completed.returncode == 0
    assert gate["metrics"]["tracking_ratio"] == pytest.approx(1.175)

    delayed = _result(0.049)
    completed, gate = _run_gate(tmp_path, "latency", zero, delayed, extra_args=nominal_arg)
    assert completed.returncode == 1
    assert gate["checks"]["tracking_ratio_at_most_1_20"] is False


def test_g1r_latency_gate_requires_nominal_result(tmp_path):
    output = tmp_path / "gate.json"
    paths = []
    for name in ("zero", "delayed"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_result(0.04)), encoding="utf-8")
        paths.append(path)
    completed = subprocess.run(
        [sys.executable, str(GATE), "latency", *(str(path) for path in paths), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert not output.exists()


def test_g1r_latency_gate_limits_zero_delay_regression_from_nominal_model(tmp_path):
    nominal = _result(0.04)
    zero = _result(0.043)
    delayed = _result(0.047)
    nominal_path = tmp_path / "nominal.json"
    nominal_path.write_text(json.dumps(nominal), encoding="utf-8")
    completed, gate = _run_gate(
        tmp_path,
        "latency",
        zero,
        delayed,
        extra_args=("--nominal-result", str(nominal_path)),
    )
    assert completed.returncode == 1
    assert gate["checks"]["zero_tracking_vs_nominal_at_most_1_05"] is False


def test_g1r_select_writes_only_a_passing_best_checkpoint(tmp_path):
    a = _result()
    b = _result()
    for name in ("forward_0.6", "backward_-0.6", "left_0.6", "right_-0.6", "yaw_left_0.6", "yaw_right_-0.6"):
        b["scenarios"][name]["tracking_rmse"] *= 1.02
    paths = []
    for name, result in (("a", a), ("b", b)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "selection.json"
    winner = tmp_path / "winner.txt"
    completed = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "select",
            *(str(path) for path in paths),
            "--checkpoints",
            "/runs/a/model_4999.pt",
            "/runs/b/model_4999.pt",
            "--winner-file",
            str(winner),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert winner.read_text(encoding="utf-8").strip() == "/runs/a/model_4999.pt"

    for result in (a, b):
        result["scenarios"]["left_0.6"]["contact"]["history_peak_n"] = 500.0
    for path, result in zip(paths, (a, b), strict=True):
        path.write_text(json.dumps(result), encoding="utf-8")
    winner.unlink()
    completed = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "select",
            *(str(path) for path in paths),
            "--checkpoints",
            "/runs/a/model_4999.pt",
            "/runs/b/model_4999.pt",
            "--winner-file",
            str(winner),
            "--output",
            str(output),
        ],
        check=False,
    )
    assert completed.returncode == 1
    assert not winner.exists()


def test_g1r_experiment_doc_separates_verified_and_pending_results():
    doc = DOC.read_text(encoding="utf-8")

    assert "已验证基线" in doc
    assert "待训练与验证" in doc
    assert "G1R-A" in doc and "G1R-B" in doc
    assert "LAT-OBS" in doc and "LAT-DROP" in doc and "LAT-ACT" in doc
    assert "LAT-COMBINED" in doc
