"""Contract checks for subset reset continuity, not a simulation capability test."""
from types import SimpleNamespace

import pytest
import torch

from reset_targets_candidate import synchronize_reset_targets


class Robot:
    def __init__(self):
        self.data = SimpleNamespace(joint_pos=torch.tensor([[.2,.93,-1.96],[-.2,-.94,1.95],[.1,.90,-1.90]]))
        self.position = torch.full((3,3), 8.)
        self.velocity = torch.full((3,3), 9.)
        self.effort = torch.full((3,3), 10.)

    def set_joint_position_target(self, value, env_ids):
        self.position[env_ids] = value

    def set_joint_velocity_target(self, value, env_ids):
        self.velocity[env_ids] = value

    def set_joint_effort_target(self, value, env_ids):
        self.effort[env_ids] = value


def test_subset_reset_uses_actual_pose_and_leaves_other_envs_untouched():
    robot = Robot()
    initial_state = robot.data.joint_pos.clone()
    synchronize_reset_targets(robot, torch.tensor([2,0]))
    torch.testing.assert_close(robot.position[[2,0]], initial_state[[2,0]])
    torch.testing.assert_close(robot.position[1], torch.full((3,),8.))
    torch.testing.assert_close(robot.velocity[[2,0]], torch.zeros(2,3))
    torch.testing.assert_close(robot.effort[[2,0]], torch.zeros(2,3))
    torch.testing.assert_close(robot.velocity[1], torch.full((3,),9.))
    torch.testing.assert_close(robot.effort[1], torch.full((3,),10.))
    torch.testing.assert_close(robot.data.joint_pos, initial_state)


def test_second_reset_follows_new_pose_not_old_target():
    robot = Robot()
    synchronize_reset_targets(robot, [0])
    robot.data.joint_pos[0] = torch.tensor([.12,.78,-1.70])
    synchronize_reset_targets(robot, [0])
    torch.testing.assert_close(robot.position[0], robot.data.joint_pos[0])


def test_empty_reset_is_noop_and_bad_shape_is_rejected():
    robot = Robot()
    synchronize_reset_targets(robot, [])
    torch.testing.assert_close(robot.position, torch.full((3,3),8.))
    with pytest.raises(ValueError, match="one-dimensional"):
        synchronize_reset_targets(robot, [[0,1]])
