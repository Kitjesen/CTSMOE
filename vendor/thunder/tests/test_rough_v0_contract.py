# Copyright (c) 2026 Shanghai Qiongpei Technology Co., Ltd.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Focused contract for the frozen Thunder rough-from-zero V0 task."""

from __future__ import annotations


def test_rough_v0_contract():
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device="cpu").app
    try:
        import gymnasium as gym
        import torch

        import doso_train.tasks  # noqa: F401
        from doso_train.tasks.terrain_inward.rough_v0_env_cfg import (
            FOOT_MATERIAL_DIM,
            JOINT_DAMPING_DIM,
            LEG_STIFFNESS_DIM,
            ROUGH_V0_CRITIC_OBS_DIM,
            DosoThunderTerrainInwardRoughV0EnvCfg,
            _mean_selected_materials,
            randomize_foot_material_with_privileged,
        )

        spec = gym.spec("DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0")
        assert spec.kwargs["env_cfg_entry_point"].endswith(
            "rough_v0_env_cfg:DosoThunderTerrainInwardRoughV0EnvCfg"
        )

        cfg = DosoThunderTerrainInwardRoughV0EnvCfg()
        assert (LEG_STIFFNESS_DIM, JOINT_DAMPING_DIM, FOOT_MATERIAL_DIM) == (12, 16, 3)
        assert ROUGH_V0_CRITIC_OBS_DIM == 274
        assert cfg.scene.terrain.max_init_terrain_level == 5
        assert tuple(cfg.scene.height_scanner.pattern_cfg.size) == (1.6, 1.0)
        assert cfg.scene.height_scanner.pattern_cfg.resolution == 0.1

        policy_terms = [
            name
            for name, term in cfg.observations.policy.__dict__.items()
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
        critic_terms = [
            name
            for name, term in cfg.observations.critic.__dict__.items()
            if hasattr(term, "func")
        ]
        assert critic_terms[-4:] == [
            "height_scan",
            "leg_stiffness_ratio",
            "joint_damping_ratio",
            "foot_material",
        ]

        command = cfg.commands.base_velocity
        assert (
            command.rel_standing_envs,
            command.rel_pure_yaw_envs,
            command.rel_pure_lateral_envs,
        ) == (0.10, 0.10, 0.10)

        events = cfg.events
        material_params = events.randomize_rigid_body_material.params
        assert material_params["static_friction_range"] == (0.3, 1.0)
        assert material_params["dynamic_friction_range"] == (0.3, 0.8)
        assert material_params["restitution_range"] == (0.0, 0.5)
        assert (
            events.randomize_foot_material_with_privileged.func
            is randomize_foot_material_with_privileged
        )
        assert events.randomize_actuator_gains.params[
            "stiffness_distribution_params"
        ] == (0.5, 2.0)
        assert events.randomize_actuator_gains.params[
            "damping_distribution_params"
        ] == (0.5, 2.0)

        assert cfg.scene.robot.actuators["hip"].stiffness == 90.0
        assert cfg.scene.robot.actuators["hip"].damping == 6.93
        assert cfg.scene.robot.actuators["wheel"].stiffness == 0.0
        assert cfg.scene.robot.actuators["wheel"].damping == 1.0

        rewards = {
            name: term.weight
            for name, term in cfg.rewards.__dict__.items()
            if hasattr(term, "weight")
        }
        assert rewards["wheel_torques_l2"] == -1.5e-3
        assert rewards["wheel_power"] == -3.0e-5
        assert rewards["feet_stumble"] == 0.0
        assert rewards["track_lin_vel_xy_exp"] == 3.75
        assert rewards["track_ang_vel_z_exp"] == 1.875
        assert len(rewards) == 21

        materials = torch.tensor(
            [
                [
                    [0.2, 0.1, 0.0],
                    [0.4, 0.3, 0.2],
                    [0.8, 0.7, 0.4],
                    [1.0, 0.9, 0.6],
                ]
            ]
        )
        selected = _mean_selected_materials(materials, [1, 2, 1], [1])
        assert torch.allclose(selected, torch.tensor([[0.6, 0.5, 0.3]]))
    finally:
        app.close()
