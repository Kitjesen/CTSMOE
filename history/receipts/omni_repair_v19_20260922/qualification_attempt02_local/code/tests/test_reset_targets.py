"""Contracts for Thunder reset-target synchronization."""

from types import SimpleNamespace

import pytest
import torch

from reset_targets import RESET_TARGETS_VERSION, install_reset_target_sync, synchronize_reset_targets


class Robot:
    def __init__(self, events: list[str] | None = None):
        self.data = SimpleNamespace(
            joint_pos=torch.tensor(
                [[0.2, 0.93, -1.96], [-0.2, -0.94, 1.95], [0.1, 0.90, -1.90]]
            )
        )
        self.position = torch.full((3, 3), 8.0)
        self.velocity = torch.full((3, 3), 9.0)
        self.effort = torch.full((3, 3), 10.0)
        self.events = events

    def _set(self, name: str, destination: torch.Tensor, value: torch.Tensor, env_ids: torch.Tensor) -> None:
        if self.events is not None:
            self.events.append(name)
        destination[env_ids] = value

    def set_joint_position_target(self, value: torch.Tensor, env_ids: torch.Tensor) -> None:
        self._set("position_target", self.position, value, env_ids)

    def set_joint_velocity_target(self, value: torch.Tensor, env_ids: torch.Tensor) -> None:
        self._set("velocity_target", self.velocity, value, env_ids)

    def set_joint_effort_target(self, value: torch.Tensor, env_ids: torch.Tensor) -> None:
        self._set("effort_target", self.effort, value, env_ids)


class Env:
    def __init__(self):
        self.events: list[str] = []
        self.robot = Robot(self.events)
        self.scene = {"robot": self.robot}
        self.reset_calls = 0

    def _reset_idx(self, env_ids: torch.Tensor) -> str:
        self.events.append("original_reset")
        self.reset_calls += 1
        self.robot.data.joint_pos[env_ids] += 0.25
        return "original-result"

    def explicit_reset(self) -> str:
        ids = torch.arange(self.robot.data.joint_pos.shape[0])
        result = self._reset_idx(ids)
        self.events.append("write_data_to_sim")
        return result


def test_natural_subset_reset_syncs_actual_pose_and_leaves_other_envs_untouched():
    env = Env()
    initial_state = env.robot.data.joint_pos.clone()
    install_reset_target_sync(env)

    result = env._reset_idx(torch.tensor([2, 0]))

    assert result == "original-result"
    expected = initial_state + torch.tensor([[0.25], [0.0], [0.25]])
    torch.testing.assert_close(env.robot.data.joint_pos, expected)
    torch.testing.assert_close(env.robot.position[[2, 0]], expected[[2, 0]])
    torch.testing.assert_close(env.robot.position[1], torch.full((3,), 8.0))
    torch.testing.assert_close(env.robot.velocity[[2, 0]], torch.zeros(2, 3))
    torch.testing.assert_close(env.robot.effort[[2, 0]], torch.zeros(2, 3))
    torch.testing.assert_close(env.robot.velocity[1], torch.full((3,), 9.0))
    torch.testing.assert_close(env.robot.effort[1], torch.full((3,), 10.0))


def test_explicit_all_reset_syncs_after_reset_and_before_scene_write():
    env = Env()
    install_reset_target_sync(env)

    result = env.explicit_reset()

    assert result == "original-result"
    assert env.events == [
        "original_reset",
        "position_target",
        "velocity_target",
        "effort_target",
        "write_data_to_sim",
    ]
    torch.testing.assert_close(env.robot.position, env.robot.data.joint_pos)
    torch.testing.assert_close(env.robot.velocity, torch.zeros(3, 3))
    torch.testing.assert_close(env.robot.effort, torch.zeros(3, 3))


def test_install_is_idempotent_auditable_and_scoped_to_one_environment():
    installed = Env()
    untouched = Env()

    first = install_reset_target_sync(installed)
    second = install_reset_target_sync(installed)
    installed._reset_idx(torch.tensor([0]))
    untouched._reset_idx(torch.tensor([0]))

    assert first == second == {"version": RESET_TARGETS_VERSION, "robot": "robot"}
    assert installed._thunder_reset_targets_installation == first
    assert installed.reset_calls == 1
    assert installed.events.count("position_target") == 1
    assert untouched.events == ["original_reset"]
    torch.testing.assert_close(untouched.robot.position, torch.full((3, 3), 8.0))


def test_second_reset_follows_new_pose_instead_of_old_target():
    robot = Robot()
    synchronize_reset_targets(robot, [0])
    robot.data.joint_pos[0] = torch.tensor([0.12, 0.78, -1.70])

    synchronize_reset_targets(robot, [0])

    torch.testing.assert_close(robot.position[0], robot.data.joint_pos[0])


def test_none_means_all_empty_is_noop_and_bad_shape_is_rejected():
    robot = Robot()
    synchronize_reset_targets(robot, [])
    torch.testing.assert_close(robot.position, torch.full((3, 3), 8.0))

    synchronize_reset_targets(robot, None)
    torch.testing.assert_close(robot.position, robot.data.joint_pos)

    with pytest.raises(ValueError, match="one-dimensional"):
        synchronize_reset_targets(robot, [[0, 1]])
