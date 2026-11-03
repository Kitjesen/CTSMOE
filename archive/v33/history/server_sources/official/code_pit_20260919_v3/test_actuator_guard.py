from types import SimpleNamespace

import pytest
import torch

from actuator_guard import (
    LIMITS,
    PEAK_PROFILE,
    PROFILE,
    SLEW_PROFILE,
    ActuatorGuard,
    actuator_violation,
    limits_for_profile,
    measured_cost,
    violation_cost,
)


class FakeScene(dict):
    def __init__(self, robot):
        super().__init__(robot=robot)
        self.update_calls = 0

    def update(self, dt):
        self.update_calls += 1
        return dt


class FakeActionManager:
    def __init__(self):
        self.actions = []

    def process_action(self, action):
        self.actions.append(action)
        return action


def make_env(profile=PROFILE):
    limits = limits_for_profile(profile)
    count = 2
    shape = (count, 2)
    data = SimpleNamespace(**{name: torch.zeros(shape) for name in (
        "applied_torque", "computed_torque", "joint_vel", "joint_acc", "joint_pos", "joint_pos_target",
    )})
    # Deliberately finite wheel bounds ensure its continuous position is masked.
    hard = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]).expand(count, -1, -1).clone()
    root_physx_view = SimpleNamespace(
        get_dof_limits=lambda: hard.clone(),
        get_dof_max_forces=lambda: torch.tensor([
            limits["leg_torque_nm"], limits["wheel_torque_nm"],
        ]).expand(count, -1).clone(),
        get_dof_max_velocities=lambda: torch.tensor([
            limits["leg_speed_rad_s"], limits["wheel_speed_rad_s"],
        ]).expand(count, -1).clone(),
    )
    robot = SimpleNamespace(
        joint_names=["fl_hip_joint", "fl_foot_joint"], data=data, root_physx_view=root_physx_view,
    )
    environment = SimpleNamespace(
        scene=FakeScene(robot), action_manager=FakeActionManager(), device="cpu",
        num_envs=count, physics_dt=0.005, cfg=SimpleNamespace(decimation=4, actuator_profile=profile),
        episode_length_buf=torch.ones(count, dtype=torch.long), _sim_step_counter=0,
    )
    ActuatorGuard(environment)
    return environment


@pytest.fixture
def env():
    return make_env()


def physics_step(env):
    env._sim_step_counter += 1
    return env.scene.update(env.physics_dt)


def test_violation_reasons_latch_each_cause_without_changing_union(env):
    data = env.scene["robot"].data
    env.action_manager.process_action(torch.zeros(2, 2))
    data.joint_vel[0, 0] = 16.
    data.joint_pos[1, 0] = 1.02
    physics_step(env)
    data.joint_vel.zero_()
    data.joint_pos.zero_()
    data.applied_torque[1, 1] = 6.01
    physics_step(env)
    assert env.actuator_guard.violation_causes["speed"].tolist() == [True, False]
    assert env.actuator_guard.violation_causes["position"].tolist() == [False, True]
    assert env.actuator_guard.violation_causes["torque"].tolist() == [False, True]
    assert env.actuator_guard.violation.tolist() == [True, True]
    env.action_manager.process_action(torch.zeros(2, 2))
    assert not env.actuator_guard.violation.any()
    assert not any(v.any() for v in env.actuator_guard.violation_causes.values())


def test_all_substeps_contribute_and_midstep_violation_remains_latched(env):
    data = env.scene["robot"].data
    env.action_manager.process_action(torch.zeros(2, 2))
    for index in range(4):
        scale = index + 1.0
        data.applied_torque.fill_(scale)
        data.computed_torque.copy_(data.applied_torque + 2.0)
        data.joint_acc.fill_(2.0 * scale)
        data.joint_vel.zero_()
        if index == 1:
            data.joint_vel[0] = torch.tensor([16.0, 31.0])
        physics_step(env)

    limits = torch.tensor([LIMITS["leg_torque_nm"], LIMITS["wheel_torque_nm"]])
    expected_torque = sum(((index / limits).square().mean() for index in (1.0, 2.0, 3.0, 4.0)))
    expected_saturation = (2.0 / limits).square().mean() * 4.0
    torch.testing.assert_close(env.actuator_guard.costs["torque_mean_square"], expected_torque.expand(2))
    torch.testing.assert_close(env.actuator_guard.costs["saturation"], expected_saturation.expand(2))
    torch.testing.assert_close(measured_cost(env, "torque_square"), torch.full((2,), 15.0))
    torch.testing.assert_close(measured_cost(env, "acceleration_square"), torch.full((2,), 60.0))
    expected_speed_cost = ((16.0 / 15.0 - 0.8) ** 2 + (31.0 / 30.0 - 0.8) ** 2) / 4.0
    torch.testing.assert_close(measured_cost(env, "speed_excess"), torch.tensor([expected_speed_cost, 0.0]))
    assert actuator_violation(env).tolist() == [True, False]
    torch.testing.assert_close(violation_cost(env), torch.tensor([1.0, 0.0]))
    assert env.actuator_guard.samples_per_joint == 8
    for joint in env.actuator_guard.snapshot()["joints"].values():
        assert joint["overspeed_samples"] == 1

    # Scene refreshes without another physics step must not double count.
    before = env.actuator_guard.snapshot()
    env.scene.update(env.physics_dt)
    assert env.actuator_guard.snapshot() == before


def test_new_control_step_clears_step_costs_and_violation_but_keeps_statistics(env):
    data = env.scene["robot"].data
    data.applied_torque.fill_(2.0)
    data.computed_torque.fill_(4.0)
    data.joint_vel[0, 1] = 40.0
    physics_step(env)
    before = env.actuator_guard.snapshot()
    streak = env.actuator_guard.streak.clone()
    assert env.actuator_guard.violation[0]

    action = torch.ones(2, 2)
    assert env.action_manager.process_action(action) is action
    assert env.action_manager.actions[-1] is action
    assert all(torch.count_nonzero(cost) == 0 for cost in env.actuator_guard.costs.values())
    assert not env.actuator_guard.violation.any()
    assert env.actuator_guard.snapshot() == before
    torch.testing.assert_close(env.actuator_guard.streak, streak)


def test_logging_reset_preserves_clipping_streak_across_log_windows(env):
    env.scene["robot"].data.computed_torque.fill_(1.0)
    physics_step(env)
    physics_step(env)
    torch.testing.assert_close(env.actuator_guard.streak, torch.full((2, 2), 0.01))
    env.actuator_guard.reset_statistics()
    assert env.actuator_guard.samples_per_joint == 0
    assert all(torch.count_nonzero(value) == 0 for value in env.actuator_guard.statistics.values())
    torch.testing.assert_close(env.actuator_guard.streak, torch.full((2, 2), 0.01))

    physics_step(env)
    snapshot = env.actuator_guard.snapshot()
    assert snapshot["samples_per_joint"] == 2
    for joint in snapshot["joints"].values():
        assert joint["max_clipped_duration_s"] == pytest.approx(0.015)
        assert joint["clipped_sample_fraction"] == 1.0


def test_episode_reset_only_clears_reset_environments_clipping_streak(env):
    env.scene["robot"].data.computed_torque.fill_(1.0)
    physics_step(env)
    physics_step(env)
    env.episode_length_buf[0] = 0
    env.action_manager.process_action(torch.zeros(2, 2))
    torch.testing.assert_close(env.actuator_guard.streak, torch.tensor([[0.0, 0.0], [0.01, 0.01]]))
    physics_step(env)
    torch.testing.assert_close(env.actuator_guard.streak, torch.tensor([[0.005, 0.005], [0.015, 0.015]]))

    # Clipping interruption clears just that joint's current continuous duration.
    env.scene["robot"].data.computed_torque[1, 0] = 0.0
    physics_step(env)
    torch.testing.assert_close(env.actuator_guard.streak, torch.tensor([[0.01, 0.01], [0.0, 0.02]]))


def test_snapshot_rms_and_clipping_fraction_use_environment_times_substep_denominator(env):
    torques = torch.tensor([
        [[3.0, 6.0], [4.0, 0.0]],
        [[0.0, 3.0], [-12.0, -6.0]],
        [[-5.0, 0.0], [0.0, -3.0]],
    ])
    clipped = torch.tensor([
        [[1.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ])
    data = env.scene["robot"].data
    for torque, clip in zip(torques, clipped, strict=True):
        data.applied_torque.copy_(torque)
        data.computed_torque.copy_(torque + clip)
        physics_step(env)

    snapshot = env.actuator_guard.snapshot()
    assert snapshot["samples_per_joint"] == 6
    expected_rms = torques.square().mean(dim=(0, 1)).sqrt()
    for index, name in enumerate(env.scene["robot"].joint_names):
        joint = snapshot["joints"][name]
        assert joint["torque_rms_nm"] == pytest.approx(expected_rms[index].item())
        assert joint["clipped_samples"] == clipped[..., index].sum().item()
        assert joint["clipped_sample_fraction"] == pytest.approx(clipped[..., index].sum().item() / 6)
        assert joint["peak_torque"] == torques[..., index].abs().max().item()


def test_applied_torque_violation_is_latched_for_either_sign_and_own_motor_limit(env):
    data = env.scene["robot"].data
    data.applied_torque[:] = torch.tensor([[35.01, 0.0], [0.0, -6.01]])
    data.computed_torque.copy_(data.applied_torque)
    physics_step(env)
    assert env.actuator_guard.violation.tolist() == [True, True]
    data.applied_torque.zero_()
    data.computed_torque.zero_()
    physics_step(env)
    assert env.actuator_guard.violation.tolist() == [True, True]
    for joint in env.actuator_guard.snapshot()["joints"].values():
        assert joint["torque_violation_samples"] == 1


def test_continuous_wheel_positions_and_targets_are_ignored_but_leg_excess_is_measured(env):
    data = env.scene["robot"].data
    data.joint_pos[:, 1] = torch.tensor([1e6, -1e6])
    data.joint_pos_target[:, 1] = torch.tensor([-1e6, 1e6])
    data.joint_pos[0, 0] = 1.02
    data.joint_pos_target[1, 0] = -1.2
    physics_step(env)
    assert env.actuator_guard.violation.tolist() == [True, False]
    joints = env.actuator_guard.snapshot()["joints"]
    assert joints["fl_foot_joint"]["max_hard_position_excess"] == 0.0
    assert joints["fl_foot_joint"]["max_target_position_excess"] == 0.0
    assert joints["fl_hip_joint"]["max_hard_position_excess"] == pytest.approx(0.02)
    assert joints["fl_hip_joint"]["max_target_position_excess"] == pytest.approx(0.2)


def test_scene_refresh_before_any_physics_step_leaves_empty_snapshot_finite(env):
    env.scene.update(env.physics_dt)
    snapshot = env.actuator_guard.snapshot()
    assert snapshot["samples_per_joint"] == 0
    for joint in snapshot["joints"].values():
        assert all(value == 0.0 for value in joint.values())


def test_slew12_keeps_physics_step_speed_and_torque_protection():
    env = make_env(SLEW_PROFILE)
    data = env.scene['robot'].data
    data.applied_torque[:] = torch.tensor([[120.0, 17.0], [120.0, 17.0]])
    data.computed_torque.copy_(data.applied_torque)
    physics_step(env)
    assert not env.actuator_guard.violation.any()
    data.joint_vel[0, 0] = 15.31
    data.joint_vel[1, 1] = 30.61
    physics_step(env)
    assert env.actuator_guard.violation.tolist() == [True, True]
    limits = env.actuator_guard.snapshot()['limits']
    assert limits == {**limits_for_profile(PEAK_PROFILE), 'leg_target_rate_rad_s': 12.0}


def test_v1_limits_remain_unchanged_and_v2_keeps_speed_and_target_constraints():
    expected_v1 = {
        "leg_torque_nm": 35.0,
        "wheel_torque_nm": 6.0,
        "leg_speed_rad_s": 15.0,
        "wheel_speed_rad_s": 30.0,
        "leg_target_rate_rad_s": 6.0,
        "speed_penalty_fraction": 0.8,
        "speed_termination_fraction": 1.02,
        "hard_position_termination_rad": 0.01,
    }
    assert LIMITS == expected_v1
    assert limits_for_profile(PROFILE) == expected_v1
    expected_v2 = {
        **expected_v1,
        "leg_torque_nm": 120.0,
        "wheel_torque_nm": 17.0,
        "leg_load_reference_nm": 35.0,
        "wheel_load_reference_nm": 6.0,
    }
    assert limits_for_profile(PEAK_PROFILE) == expected_v2
    independent = limits_for_profile(PEAK_PROFILE)
    independent["leg_torque_nm"] = -1.0
    assert limits_for_profile(PEAK_PROFILE) == expected_v2
    assert limits_for_profile(PROFILE) == expected_v1


def test_v2_applies_soft_load_penalty_above_reference_without_false_hard_limit_violation():
    env = make_env(PEAK_PROFILE)
    guard = env.actuator_guard
    torch.testing.assert_close(guard.torque_limit, torch.tensor([120.0, 17.0]))
    torch.testing.assert_close(guard.load_reference, torch.tensor([35.0, 6.0]))
    torch.testing.assert_close(guard.speed_limit, torch.tensor([15.0, 30.0]))
    data = env.scene["robot"].data
    data.applied_torque[:] = torch.tensor([[70.0, -12.0], [-70.0, 12.0]])
    data.computed_torque.copy_(data.applied_torque)
    physics_step(env)
    assert not guard.violation.any()
    torch.testing.assert_close(guard.costs["torque_mean_square"], torch.tensor([4.0, 4.0]))
    assert torch.count_nonzero(guard.costs["saturation"]) == 0
    assert guard.snapshot()["profile"] == PEAK_PROFILE
    assert guard.snapshot()["limits"] == limits_for_profile(PEAK_PROFILE)
    assert all(joint["torque_violation_samples"] == 0 for joint in guard.snapshot()["joints"].values())


def test_v2_peak_torque_is_allowed_but_exceeding_each_motor_hard_limit_is_flagged():
    env = make_env(PEAK_PROFILE)
    data = env.scene["robot"].data
    data.applied_torque[:] = torch.tensor([[120.0, -17.0], [-120.0, 17.0]])
    data.computed_torque.copy_(data.applied_torque)
    physics_step(env)
    assert not env.actuator_guard.violation.any()
    data.applied_torque[:] = torch.tensor([[120.01, 0.0], [0.0, -17.01]])
    data.computed_torque.copy_(data.applied_torque)
    physics_step(env)
    assert env.actuator_guard.violation.tolist() == [True, True]
    for joint in env.actuator_guard.snapshot()["joints"].values():
        assert joint["torque_violation_samples"] == 1


def test_v2_saturation_cost_uses_hard_limits_and_retains_v1_speed_violation_thresholds():
    env = make_env(PEAK_PROFILE)
    data = env.scene["robot"].data
    data.computed_torque[:] = torch.tensor([[120.0, -17.0], [-120.0, 17.0]])
    data.joint_vel[:] = torch.tensor([[15.31, 0.0], [0.0, -30.61]])
    physics_step(env)
    torch.testing.assert_close(env.actuator_guard.costs["saturation"], torch.ones(2))
    assert env.actuator_guard.violation.tolist() == [True, True]
