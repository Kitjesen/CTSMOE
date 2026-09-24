import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from default_pose import PROFILE, configure, contract, resolve_profile, validate_runtime
from policy_contract import export_contract, load_student_export
from train_scratch import build_command


def test_reference_matches_flat_handoff_exactly():
    reference = Path(__file__).resolve().parents[3] / "flat/reports/20260923-flat-handoff/DEFAULT_POSE.json"
    flat = json.loads(reference.read_text(encoding="utf-8"))
    joints = contract(PROFILE)["joint_position_rad"]
    assert [joints[n] for n in flat["action_joint_names"]] == (
        flat["default_leg_position"] + flat["default_wheel_position"])


def test_config_changes_pose_not_limits_gains_height_or_shared_asset():
    init = NS(joint_pos={".*_hip_joint": 0.2}, joint_vel={".*": 0.1}, pos=(0, 0, .46))
    cfg = NS(scene=NS(robot=NS(init_state=init, actuators={"kp": 120, "kd": 8})),
             actions=NS(joint_pos=NS(use_default_offset=True, scale=.25, clip=(-.369, .369))))
    before = copy.deepcopy(cfg)
    configure(cfg, PROFILE)
    assert init.joint_pos == {".*_hip_joint": .2}
    assert cfg.scene.robot.init_state.pos == before.scene.robot.init_state.pos
    assert cfg.scene.robot.actuators == before.scene.robot.actuators
    assert vars(cfg.actions.joint_pos) == vars(before.actions.joint_pos)
    assert len(cfg.scene.robot.init_state.joint_pos) == 16
    assert cfg.scene.robot.init_state.joint_vel == {".*": 0.0}


def test_checkpoint_inherits_pose_and_rejects_silent_change_or_corruption():
    saved = {"default_pose_contract": contract(PROFILE)}
    assert resolve_profile(None, saved) == PROFILE
    assert resolve_profile(None, {}) == "asset_original"
    assert resolve_profile(PROFILE, None) == PROFILE
    with pytest.raises(ValueError):
        resolve_profile(PROFILE, {})
    saved["default_pose_contract"]["joint_position_rad"]["fr_hip_joint"] = .12
    with pytest.raises(ValueError):
        resolve_profile(None, saved)


def test_runtime_checks_names_instead_of_assuming_joint_order():
    expected = contract(PROFILE)["joint_position_rad"]
    names = list(reversed(expected))
    q = torch.tensor([[expected[n] for n in names]]).repeat(3, 1)
    env = NS(scene={"robot": NS(joint_names=names, data=NS(
        default_joint_pos=q, default_joint_vel=torch.zeros_like(q)))})
    assert validate_runtime(env, PROFILE)["default_positions_verified"]
    q[:, names.index("fr_hip_joint")] = .12
    with pytest.raises(AssertionError):
        validate_runtime(env, PROFILE)


def test_student_export_cannot_silently_use_old_default(tmp_path):
    policy = torch.jit.trace(torch.nn.Linear(2, 1), torch.ones(1, 2))
    path = tmp_path / "student.pt"
    policy.save(str(path), _extra_files={
        "actuator_contract.json": json.dumps(export_contract("original")),
        "default_pose.json": json.dumps(contract(PROFILE))})
    with pytest.raises(ValueError, match="pose mismatch"):
        load_student_export(path)
    loaded, _ = load_student_export(path, expected_pose=PROFILE)
    torch.testing.assert_close(policy(torch.ones(1, 2)), loaded(torch.ones(1, 2)))


def scratch_args(tmp_path):
    return ["--root", str(tmp_path), "--output", str(tmp_path / "new_run"),
            "--updates", "8", "--num-envs", "64", "--terrain-course", "original8"]


def test_scratch_requires_explicit_budget_and_rejects_any_resume(tmp_path):
    args = scratch_args(tmp_path)
    _, command = build_command(args)
    assert "--resume" not in command
    assert command[command.index("--default-pose") + 1] == PROFILE
    for extra in (["--resume", "old.pt"], ["--mixed-pit"], ["--updates", "0"]):
        with pytest.raises(SystemExit):
            build_command(args + extra)
    with pytest.raises(SystemExit):
        build_command(args[:4] + args[6:])
