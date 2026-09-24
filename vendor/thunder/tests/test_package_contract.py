# Copyright (c) 2026 Shanghai Qiongpei Technology Co., Ltd.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Contract tests for the standalone DOSO Thunder extension."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import gymnasium as gym
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "source" / "doso_train" / "doso_train"
TASK_IDS = (
    "DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0",
    "DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0",
)


def test_gym_registration_is_lightweight_and_complete():
    import doso_train

    assert doso_train.__version__ == "0.1.0"

    for task_id in TASK_IDS:
        spec = gym.spec(task_id)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert set(spec.kwargs) == {
            "env_cfg_entry_point",
            "rsl_rl_cfg_entry_point",
            "cusrl_cfg_entry_point",
        }
        assert spec.kwargs["env_cfg_entry_point"].startswith(
            "doso_train.tasks.terrain_inward."
        )
        assert spec.kwargs["rsl_rl_cfg_entry_point"].startswith(
            "doso_train.tasks.terrain_inward.agents.rsl_rl_ppo_cfg:"
        )
        assert spec.kwargs["cusrl_cfg_entry_point"].startswith(
            "doso_train.tasks.terrain_inward.agents.cusrl_ppo_cfg:"
        )


def test_asset_payload_is_exact_and_self_contained():
    asset_root = PACKAGE_ROOT / "data" / "Robots" / "DOSO" / "thunder"
    urdfs = list((asset_root / "urdf").glob("*.urdf"))
    meshes = list((asset_root / "meshes").glob("*.STL"))
    assert [path.name for path in urdfs] == ["thunder_v4.urdf"]
    assert len(meshes) == 21

    root = ET.parse(urdfs[0]).getroot()
    mesh_refs = [node.attrib["filename"] for node in root.findall(".//mesh")]
    assert len(mesh_refs) == 21
    assert len(set(mesh_refs)) == 21
    assert all((urdfs[0].parent / ref).resolve().is_file() for ref in mesh_refs)
    assert len(root.findall("link")) == 21
    assert len(root.findall("joint")) == 20


def test_wheel_contains_runtime_assets(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
            str(PROJECT_ROOT / "source" / "doso_train"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(tmp_path.glob("doso_train-0.1.0-*.whl"))
    assert len(wheels) == 1, wheels
    with ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        assert len([name for name in names if name.lower().endswith(".urdf")]) == 1
        assert len([name for name in names if name.lower().endswith(".stl")]) == 21
        assert "doso_train/tasks/terrain_inward/env_cfg.py" in names
        assert "doso_train/tasks/terrain_inward/mdp.py" in names
        assert "doso_train/tasks/terrain_inward/rough_v0_env_cfg.py" in names


def test_source_has_no_robot_lab_runtime_dependency():
    python_files = list((PROJECT_ROOT / "source").rglob("*.py")) + list(
        (PROJECT_ROOT / "scripts").rglob("*.py")
    )
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("robot_lab") for alias in node.names
                ), path
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("robot_lab"), path

    cusrl_play = (
        PROJECT_ROOT / "scripts" / "reinforcement_learning" / "cusrl" / "play.py"
    ).read_text(encoding="utf-8")
    assert "rl_utils" not in cusrl_play
    assert "camera_follow" not in cusrl_play
    assert "isaaclab.devices" not in cusrl_play


def _load_kit_contract():
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device="cpu").app
    try:
        import doso_train.tasks  # noqa: F401
        from doso_train.assets import DOSO_THUNDER_V4_CFG
        from doso_train.tasks.terrain_inward import mdp
        from doso_train.tasks.terrain_inward.agents.cusrl_ppo_cfg import (
            DosoThunderTerrainInwardFlatTrainerCfg,
            DosoThunderTerrainInwardRoughTrainerCfg,
        )
        from doso_train.tasks.terrain_inward.agents.rsl_rl_ppo_cfg import (
            DosoThunderTerrainInwardFlatPPORunnerCfg,
            DosoThunderTerrainInwardRoughPPORunnerCfg,
        )
        from doso_train.tasks.terrain_inward.env_cfg import (
            FLAT_CRITIC_OBS_DIM,
            HEIGHT_SCAN_DIM,
            JOINT_NAMES,
            LEG_JOINT_NAMES,
            POLICY_OBS_DIM,
            ROUGH_CRITIC_OBS_DIM,
            WHEEL_JOINT_NAMES,
            DosoThunderTerrainInwardFlatEnvCfg,
            DosoThunderTerrainInwardRoughEnvCfg,
        )

        return app, {
            "asset": DOSO_THUNDER_V4_CFG,
            "mdp": mdp,
            "flat": DosoThunderTerrainInwardFlatEnvCfg(),
            "rough": DosoThunderTerrainInwardRoughEnvCfg(),
            "rsl_flat": DosoThunderTerrainInwardFlatPPORunnerCfg(),
            "rsl_rough": DosoThunderTerrainInwardRoughPPORunnerCfg(),
            "cusrl_flat": DosoThunderTerrainInwardFlatTrainerCfg(),
            "cusrl_rough": DosoThunderTerrainInwardRoughTrainerCfg(),
            "dims": (
                POLICY_OBS_DIM,
                FLAT_CRITIC_OBS_DIM,
                HEIGHT_SCAN_DIM,
                ROUGH_CRITIC_OBS_DIM,
            ),
            "joints": (LEG_JOINT_NAMES, WHEEL_JOINT_NAMES, JOINT_NAMES),
        }
    except BaseException:
        app.close()
        raise


def _assert_observation_action_and_command_contract(kit_contract):
    flat = kit_contract["flat"]
    rough = kit_contract["rough"]
    mdp = kit_contract["mdp"]
    legs, wheels, joints = kit_contract["joints"]
    assert kit_contract["dims"] == (53, 56, 187, 243)
    policy_dim = 3 + 3 + 3 + len(legs) + len(joints) + len(joints)
    flat_critic_dim = 3 + policy_dim
    assert policy_dim == 53
    assert flat_critic_dim == 56
    assert flat_critic_dim + kit_contract["dims"][2] == 243

    policy_terms = [
        name
        for name, term in flat.observations.policy.__dict__.items()
        if hasattr(term, "func")
    ]
    assert policy_terms == [
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    assert flat.observations.policy.base_ang_vel.scale == 0.25
    assert flat.observations.policy.joint_pos.scale == 1.0
    assert flat.observations.policy.joint_vel.scale == 0.05
    assert (
        flat.observations.policy.base_ang_vel.noise.n_min,
        flat.observations.policy.base_ang_vel.noise.n_max,
    ) == (-0.2, 0.2)
    assert (
        flat.observations.policy.projected_gravity.noise.n_min,
        flat.observations.policy.projected_gravity.noise.n_max,
    ) == (-0.05, 0.05)
    assert (
        flat.observations.policy.joint_pos.noise.n_min,
        flat.observations.policy.joint_pos.noise.n_max,
    ) == (-0.01, 0.01)
    assert (
        flat.observations.policy.joint_vel.noise.n_min,
        flat.observations.policy.joint_vel.noise.n_max,
    ) == (-1.5, 1.5)
    assert flat.observations.policy.enable_corruption is True
    assert flat.observations.policy.joint_pos.params["asset_cfg"].joint_names == legs
    assert flat.observations.policy.joint_vel.params["asset_cfg"].joint_names == joints
    assert flat.observations.critic.height_scan is None
    assert rough.observations.critic.height_scan is not None
    assert rough.observations.critic.enable_corruption is False
    assert rough.observations.critic.base_lin_vel.noise is None

    assert flat.actions.joint_pos.joint_names == legs
    assert flat.actions.joint_vel.joint_names == wheels
    assert flat.actions.joint_pos.scale == {
        ".*_hip_joint": 0.125,
        "^(?!.*_hip_joint).*": 0.25,
    }
    assert flat.actions.joint_vel.scale == 5.0
    assert flat.actions.joint_pos.clip == {".*": (-100.0, 100.0)}
    assert flat.actions.joint_vel.clip == {".*": (-100.0, 100.0)}
    assert (
        flat.actions.joint_pos.preserve_order and flat.actions.joint_vel.preserve_order
    )
    assert (
        flat.actions.joint_pos.use_default_offset
        and flat.actions.joint_vel.use_default_offset
    )

    command = flat.commands.base_velocity
    assert command.resampling_time_range == (10.0, 10.0)
    assert command.rel_standing_envs == 0.15
    assert command.rel_pure_yaw_envs == 0.15
    assert command.rel_pure_lateral_envs == 0.10
    assert command.pure_command_min_magnitude == 0.2
    assert command.rel_heading_envs == 1.0
    assert command.heading_command and command.heading_control_stiffness == 0.5
    assert command.ranges.lin_vel_x == (-1.0, 1.0)
    assert command.ranges.lin_vel_y == (-1.0, 1.0)
    assert command.ranges.ang_vel_z == (-1.0, 1.0)
    assert command.ranges.heading == (-math.pi, math.pi)
    assert command.class_type is mdp.UniformThresholdVelocityCommand
    assert mdp.COMMAND_DEADZONE == 0.2
    assert mdp.PIT_FORWARD_SPEED_RANGE == (0.3, 0.6)


def _assert_reward_event_termination_and_terrain_contract(kit_contract):
    flat = kit_contract["flat"]
    rough = kit_contract["rough"]
    expected_rewards = {
        "lin_vel_z_l2": -2.0,
        "ang_vel_xy_l2": -0.05,
        "flat_orientation_l2": 0.0,
        "joint_torques_l2": -3.0e-5,
        "joint_acc_l2": -1.0e-7,
        "joint_acc_wheel_l2": -2.5e-10,
        "joint_pos_limits": -5.0,
        "joint_power": -1.0e-5,
        "wheel_torques_l2": -1.5e-3,
        "wheel_power": -3.0e-5,
        "standing_lin_vel_xy_l2": -2.0,
        "standing_ang_vel_z_l2": -0.5,
        "standing_wheel_vel_l2": -0.01,
        "pure_yaw_lin_vel_xy_l2": -2.0,
        "pure_yaw_wheel_mirror_l2": 0.0,
        "stand_still": -2.0,
        "joint_pos_penalty": -1.0,
        "hip_abs_limit_l2": 0.0,
        "joint_mirror": -0.05,
        "action_rate_l2": -0.01,
        "wheel_contact_slip_l2": 0.0,
        "mode_swing_clearance_l2": 0.0,
        "minimum_wheel_contacts_l2": 0.0,
        "undesired_contacts": -1.0,
        "contact_forces": -1.0e-2,
        "touchdown_velocity_l2": -0.2,
        "track_lin_vel_xy_exp": 3.75,
        "track_ang_vel_z_exp": 1.875,
        "upward": 3.0,
    }
    actual_rewards = {
        name: term.weight
        for name, term in rough.rewards.__dict__.items()
        if hasattr(term, "weight")
    }
    assert actual_rewards == expected_rewards

    assert rough.rewards.contact_forces.params["threshold"] == 450.0
    touchdown = rough.rewards.touchdown_velocity_l2
    assert touchdown.params["velocity_threshold"] == 0.15
    assert touchdown.params["sensor_cfg"].body_names == [".*_foot_Link"]
    assert touchdown.params["asset_cfg"].body_names == [".*_foot_Link"]

    slip = rough.rewards.wheel_contact_slip_l2
    expected_collision_center_y = {
        "fr_foot_Link": -0.01631,
        "fl_foot_Link": 0.01631,
        "rr_foot_Link": -0.01631,
        "rl_foot_Link": 0.01631,
    }
    assert kit_contract["mdp"].WHEEL_COLLISION_CENTER_Y == expected_collision_center_y
    assert slip.func is kit_contract["mdp"].wheel_contact_slip_l2
    assert slip.params["wheel_radius"] == 0.093
    assert slip.params["velocity_deadzone"] == 0.02
    assert slip.params["max_slip_speed"] == 1.0
    assert slip.params["collision_center_y"] == expected_collision_center_y
    assert slip.params["sensor_cfg"].body_names == [".*_foot_Link"]
    assert slip.params["asset_cfg"].body_names == [".*_foot_Link"]

    clearance = rough.rewards.mode_swing_clearance_l2
    assert clearance.params["lateral_clearance"] == 0.04
    assert clearance.params["yaw_clearance"] == 0.03
    assert clearance.params["wheel_radius"] == 0.093
    assert clearance.params["collision_center_y"] == slip.params["collision_center_y"]

    support = rough.rewards.minimum_wheel_contacts_l2
    assert support.params["minimum_contacts"] == 2
    assert support.params["sensor_cfg"].body_names == [".*_foot_Link"]

    mdp = kit_contract["mdp"]
    first_contact = mdp.torch.tensor([[True, True, False, True]])
    previous_foot_vel_z = mdp.torch.tensor([[-0.10, -0.40, -1.00, 0.20]])
    actual_touchdown_penalty = mdp._touchdown_velocity_excess_l2(
        first_contact,
        previous_foot_vel_z,
        velocity_threshold=0.15,
    )
    assert mdp.torch.allclose(
        actual_touchdown_penalty,
        mdp.torch.tensor([0.25**2]),
    )

    reordered_centers = mdp._wheel_collision_centers_b(
        ["fl_foot_Link", "fr_foot_Link", "rl_foot_Link", "rr_foot_Link"],
        mdp.WHEEL_COLLISION_CENTER_Y,
        "cpu",
    )
    assert mdp.torch.equal(
        reordered_centers[0, :, 1],
        mdp.torch.tensor([0.01631, -0.01631, 0.01631, -0.01631]),
    )

    contact = mdp.torch.tensor([[True, True]])
    body_lin_vel_w = mdp.torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    body_ang_vel_w = mdp.torch.tensor([[[0.0, 1.0 / 0.093, 0.0], [0.0, 0.0, 0.0]]])
    body_quat_w = mdp.torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    collision_center_b = mdp.torch.tensor([[[0.0, -0.01631, 0.0], [0.0, 0.01631, 0.0]]])
    slip_penalty = mdp._wheel_contact_point_slip_l2(
        contact,
        body_lin_vel_w,
        body_ang_vel_w,
        body_quat_w,
        collision_center_b,
        wheel_radius=0.093,
        velocity_deadzone=0.02,
        max_slip_speed=1.0,
    )
    assert mdp.torch.allclose(slip_penalty, mdp.torch.tensor([0.98**2]))

    tilt = 0.28
    tilted_quat = mdp.torch.tensor(
        [[[math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0]]]
    )
    tilted_center = mdp.torch.tensor([[[0.0, -0.01631, 0.0]]])
    offset_w = mdp._wheel_contact_offset_w(tilted_quat, tilted_center, 0.093)
    local_axis = mdp.torch.tensor([[[0.0, 1.0, 0.0]]])
    angular_velocity = mdp.quat_apply(tilted_quat, local_axis) / 0.093
    ideal_link_velocity = -mdp.torch.cross(angular_velocity, offset_w, dim=-1)
    ideal_penalty = mdp._wheel_contact_point_slip_l2(
        mdp.torch.tensor([[True]]),
        ideal_link_velocity,
        angular_velocity,
        tilted_quat,
        tilted_center,
        wheel_radius=0.093,
        velocity_deadzone=0.02,
        max_slip_speed=1.0,
    )
    assert mdp.torch.allclose(ideal_penalty, mdp.torch.zeros(1), atol=1.0e-6)

    lateral_slip = ideal_link_velocity + mdp.torch.tensor([[[0.1, 0.0, 0.0]]])
    lateral_penalty = mdp._wheel_contact_point_slip_l2(
        mdp.torch.tensor([[True]]),
        lateral_slip,
        angular_velocity,
        tilted_quat,
        tilted_center,
        wheel_radius=0.093,
        velocity_deadzone=0.02,
        max_slip_speed=1.0,
    )
    assert mdp.torch.allclose(lateral_penalty, mdp.torch.tensor([0.08**2]), atol=1.0e-6)

    command_term = SimpleNamespace(
        is_standing_env=mdp.torch.tensor([True, False]),
        is_pure_yaw_env=mdp.torch.tensor([False, True]),
        is_pure_lateral_env=mdp.torch.tensor([False, False]),
    )
    mode_env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_lin_vel_b=mdp.torch.tensor([[3.0, 4.0, 0.0], [1.0, 2.0, 0.0]]),
                    root_ang_vel_b=mdp.torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]),
                    joint_vel=mdp.torch.tensor(
                        [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 3.0, 4.0]]
                    ),
                )
            )
        },
    )
    assert mdp.torch.equal(
        mdp.standing_lin_vel_xy_l2(mode_env, "base_velocity"),
        mdp.torch.tensor([25.0, 0.0]),
    )
    assert mdp.torch.equal(
        mdp.standing_ang_vel_z_l2(mode_env, "base_velocity"),
        mdp.torch.tensor([4.0, 0.0]),
    )
    assert mdp.torch.equal(
        mdp.standing_wheel_vel_l2(
            mode_env,
            "base_velocity",
            SimpleNamespace(name="robot", joint_ids=[0, 1]),
        ),
        mdp.torch.tensor([5.0, 0.0]),
    )
    assert mdp.torch.equal(
        mdp.pure_yaw_lin_vel_xy_l2(mode_env, "base_velocity"),
        mdp.torch.tensor([0.0, 5.0]),
    )
    assert mdp.torch.equal(
        mdp.pure_yaw_wheel_mirror_l2(
            mode_env,
            "base_velocity",
            SimpleNamespace(name="robot", joint_ids=[0, 1, 2, 3]),
        ),
        mdp.torch.tensor([0.0, 0.25]),
    )

    mirror_modes = SimpleNamespace(
        is_standing_env=mdp.torch.tensor([True, False, False, False]),
        is_pure_yaw_env=mdp.torch.tensor([False, True, False, False]),
        is_pure_lateral_env=mdp.torch.tensor([False, False, True, False]),
    )
    mirror_env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        command_manager=SimpleNamespace(get_term=lambda _name: mirror_modes),
    )
    assert mdp.torch.equal(
        mdp.command_mode_scale(
            mirror_env,
            "base_velocity",
            standing_scale=1.0,
            pure_yaw_scale=0.5,
            pure_lateral_scale=0.0,
            mixed_scale=1.0,
        ),
        mdp.torch.tensor([1.0, 0.5, 0.0, 1.0]),
    )

    base_mass = mdp.torch.tensor([10.0, 10.0])
    base_com = mdp.torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
    base_inertia = mdp.torch.diag_embed(
        mdp.torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    )
    total_mass, total_com, total_inertia = mdp._compose_point_mass_payload(
        base_mass,
        base_com,
        base_inertia,
        mdp.torch.tensor([0.0, 2.0]),
        mdp.torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
    )
    assert total_mass.tolist() == [10.0, 12.0]
    assert mdp.torch.equal(total_com[0], base_com[0])
    assert mdp.torch.equal(total_inertia[0], base_inertia[0])
    assert mdp.torch.allclose(total_inertia, total_inertia.transpose(-1, -2))
    assert mdp.torch.all(mdp.torch.linalg.eigvalsh(total_inertia) > 0.0)

    class _FakeContactSensor:
        def __init__(self):
            self.first_contact = mdp.torch.tensor(
                [
                    [True, False, False, False],
                    [False, False, False, False],
                ]
            )

        def compute_first_contact(self, _step_dt):
            return self.first_contact

    class _FakeScene:
        def __init__(self, asset, sensor):
            self._asset = asset
            self.sensors = {"contact_forces": sensor}

        def __getitem__(self, name):
            assert name == "robot"
            return self._asset

    current_foot_vel_z = mdp.torch.tensor(
        [
            [-0.10, -0.20, 0.00, 0.10],
            [-0.30, -0.40, -0.50, -0.60],
        ]
    )
    body_lin_vel_w = mdp.torch.zeros(2, 4, 3)
    body_lin_vel_w[:, :, 2] = current_foot_vel_z
    sensor = _FakeContactSensor()
    env = SimpleNamespace(
        scene=_FakeScene(
            SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=body_lin_vel_w)),
            sensor,
        ),
        step_dt=0.02,
    )
    asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3])
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3])
    term = object.__new__(mdp.touchdown_velocity_l2)
    term._previous_foot_vel_z = mdp.torch.tensor(
        [
            [-0.40, -0.70, -1.00, 0.20],
            [-0.30, -0.40, -0.50, -0.60],
        ]
    )

    first_penalty = term(env, 0.15, asset_cfg, sensor_cfg)
    assert mdp.torch.allclose(first_penalty, mdp.torch.tensor([0.25**2, 0.0]))
    assert mdp.torch.equal(term._previous_foot_vel_z, current_foot_vel_z)

    sensor.first_contact.zero_()
    assert mdp.torch.equal(term(env, 0.15, asset_cfg, sensor_cfg), mdp.torch.zeros(2))

    previous_second_env = term._previous_foot_vel_z[1].clone()
    term.reset(mdp.torch.tensor([0]))
    assert mdp.torch.equal(term._previous_foot_vel_z[0], mdp.torch.zeros(4))
    assert mdp.torch.equal(term._previous_foot_vel_z[1], previous_second_env)

    events = rough.events
    assert events.randomize_rigid_body_material.params["static_friction_range"] == (
        0.25,
        1.2,
    )
    assert events.randomize_rigid_body_material.params["dynamic_friction_range"] == (
        0.2,
        1.0,
    )
    assert events.randomize_rigid_body_material.params["restitution_range"] == (
        0.0,
        0.2,
    )
    assert events.randomize_rigid_body_material.params["num_buckets"] == 64
    assert events.randomize_rigid_body_material.params["make_consistent"] is True
    payload = events.randomize_base_payload
    assert payload.func is mdp.randomize_base_payload
    assert payload.params["payload_mass_range"] == (0.0, 12.0)
    assert payload.params["payload_zero_probability"] == 0.35
    assert payload.params["payload_position_range"] == {
        "x": (-0.15, 0.15),
        "y": (-0.10, 0.10),
        "z": (0.05, 0.20),
    }
    assert payload.params["base_mass_scale_range"] == (0.95, 1.05)
    assert payload.params["base_com_bias_range"] == {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (-0.02, 0.02),
    }
    assert not hasattr(events, "randomize_rigid_body_mass_base")
    assert not hasattr(events, "randomize_com_positions")
    assert events.randomize_rigid_body_mass_others.params[
        "mass_distribution_params"
    ] == (0.7, 1.3)
    assert events.randomize_rigid_body_mass_others.params["operation"] == "scale"
    assert events.randomize_rigid_body_mass_others.params["recompute_inertia"] is True
    assert events.randomize_apply_external_force_torque.params["force_range"] == (
        -30.0,
        30.0,
    )
    assert events.randomize_apply_external_force_torque.params["torque_range"] == (
        -10.0,
        10.0,
    )
    assert events.randomize_reset_joints.params == {
        "position_range": (1.0, 1.0),
        "velocity_range": (0.0, 0.0),
    }
    assert events.randomize_actuator_gains.params["stiffness_distribution_params"] == (
        0.8,
        1.2,
    )
    assert events.randomize_actuator_gains.params["damping_distribution_params"] == (
        0.8,
        1.2,
    )
    assert events.randomize_actuator_gains.params["operation"] == "scale"
    assert events.randomize_actuator_gains.params["distribution"] == "uniform"
    assert events.randomize_reset_base.params["pose_range"] == {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.2),
        "roll": (-3.14, 3.14),
        "pitch": (-3.14, 3.14),
        "yaw": (-3.14, 3.14),
    }
    assert events.randomize_reset_base.params["velocity_range"] == {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    }
    assert events.randomize_push_robot.interval_range_s == (10.0, 15.0)
    assert events.randomize_push_robot.params == {
        "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
    }

    termination_names = {
        name
        for name, term in rough.terminations.__dict__.items()
        if hasattr(term, "func") and term is not None
    }
    assert termination_names == {"time_out", "terrain_out_of_bounds"}
    assert rough.terminations.terrain_out_of_bounds.params["distance_buffer"] == 3.0
    assert rough.scene.terrain.terrain_type == "generator"
    assert rough.scene.height_scanner is not None
    assert tuple(rough.scene.height_scanner.pattern_cfg.size) == (1.6, 1.0)
    assert rough.scene.height_scanner.pattern_cfg.resolution == 0.1
    scan_size = rough.scene.height_scanner.pattern_cfg.size
    scan_resolution = rough.scene.height_scanner.pattern_cfg.resolution
    assert (round(scan_size[0] / scan_resolution) + 1) * (
        round(scan_size[1] / scan_resolution) + 1
    ) == 187
    assert rough.curriculum.terrain_levels is not None
    assert flat.scene.terrain.terrain_type == "plane"
    assert flat.scene.terrain.terrain_generator is None
    assert flat.scene.height_scanner is None
    assert flat.curriculum.terrain_levels is None
    assert rough.scene.terrain.physics_material.restitution == 1.0
    assert rough.scene.terrain.physics_material.restitution_combine_mode == "multiply"
    assert rough.sim.dt == 0.005
    assert rough.decimation == 4
    assert rough.episode_length_s == 20.0
    assert rough.sim.physx.gpu_max_rigid_patch_count == 10 * 2**15


def _assert_asset_and_runner_contract(kit_contract):
    asset = kit_contract["asset"]
    expected_joint_pos = {
        "fr_hip_joint": -0.20,
        "fr_thigh_joint": -0.93,
        "fr_calf_joint": 1.96,
        "fl_hip_joint": 0.20,
        "fl_thigh_joint": 0.93,
        "fl_calf_joint": -1.96,
        "rr_hip_joint": 0.20,
        "rr_thigh_joint": 0.93,
        "rr_calf_joint": -1.96,
        "rl_hip_joint": -0.20,
        "rl_thigh_joint": -0.93,
        "rl_calf_joint": 1.96,
        ".*_foot_joint": 0.0,
    }
    assert asset.init_state.pos == (0.0, 0.0, 0.46)
    assert asset.init_state.joint_pos == expected_joint_pos
    for name in ("hip", "thigh", "calf"):
        actuator = asset.actuators[name]
        assert actuator.effort_limit == 120.0
        assert actuator.saturation_effort == 120.0
        assert actuator.velocity_limit == 17.48
        assert actuator.stiffness == 120.0
        assert actuator.damping == 8.0
    wheel = asset.actuators["wheel"]
    assert wheel.effort_limit_sim == 17.0
    assert wheel.velocity_limit_sim == 44.0
    assert wheel.stiffness == 0.0
    assert wheel.damping == 1.0

    assert kit_contract["rsl_flat"].max_iterations == 5000
    assert kit_contract["rsl_rough"].max_iterations == 20000
    assert (
        kit_contract["rsl_flat"].experiment_name == "doso_thunder_terrain_inward_flat"
    )
    assert (
        kit_contract["rsl_rough"].experiment_name == "doso_thunder_terrain_inward_rough"
    )
    assert kit_contract["rsl_rough"].num_steps_per_env == 24
    assert kit_contract["rsl_rough"].save_interval == 100
    assert kit_contract["cusrl_flat"].max_iterations == 5000
    assert kit_contract["cusrl_rough"].max_iterations == 20000
    assert (
        kit_contract["cusrl_flat"].experiment_name == "doso_thunder_terrain_inward_flat"
    )
    assert (
        kit_contract["cusrl_rough"].experiment_name
        == "doso_thunder_terrain_inward_rough"
    )
    assert kit_contract["cusrl_rough"].save_interval == 100


def test_kit_contract_in_isolated_process():
    if os.environ.get("DOSO_SKIP_KIT_CONTRACT") == "1":
        pytest.skip("Isaac Sim contract explicitly skipped on an unhealthy Kit host")
    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--kit-contract"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("DOSO_KIT_TEST_TIMEOUT", "180")),
        check=False,
    )
    output_tail = (completed.stdout + "\n" + completed.stderr)[-8000:]
    assert completed.returncode == 0, output_tail
    assert "KIT_CONTRACT_OK" in completed.stdout, output_tail


if __name__ == "__main__" and "--kit-contract" in sys.argv:
    simulation_app, contract = _load_kit_contract()
    try:
        _assert_observation_action_and_command_contract(contract)
        _assert_reward_event_termination_and_terrain_contract(contract)
        _assert_asset_and_runner_contract(contract)
        print("KIT_CONTRACT_OK", flush=True)
    finally:
        simulation_app.close()
