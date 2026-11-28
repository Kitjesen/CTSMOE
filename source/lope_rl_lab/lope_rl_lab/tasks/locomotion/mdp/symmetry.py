"""Segmented CTS mirror for Lope's 12 revolute joints.

Batch ordering comes from the pinned Go2 symmetry helper. Joint signs do not:
Go2 flips hips only, and that map does not fix Lope's standing thighs.
"""
import torch

from go2_rl_lab.utils.symmetry import Go2Symmetry

from lope_rl_lab.assets.lope_cfg import (
    FOOT_PERM,
    JOINT_NAMES,
    JOINT_PERM,
    JOINT_SIGN,
    STANDING_POSE,
)
from lope_rl_lab.tasks.locomotion.config import (
    CRITIC_DIM,
    CRITIC_PROPRIO_DIM,
    FOOT_FORCE_DIM,
    FRAME_DIM,
    HISTORY_LENGTH,
    POLICY_DIM,
)
from lope_rl_lab.tasks.locomotion.mdp.observations import tensor
from lope_rl_lab.utils.symmetry import mirror_joints, reflected_ray_indices


class LopeSymmetryMapper:
    def __init__(self, env):
        base = getattr(env, "unwrapped", env)
        robot = base.scene["robot"]
        defaults = tensor(robot.data.default_joint_pos)[0]
        pose = [float(defaults[robot.joint_names.index(name)]) for name in JOINT_NAMES]
        expected = [STANDING_POSE[name] for name in JOINT_NAMES]
        if not torch.allclose(torch.tensor(mirror_joints(pose)), torch.tensor(expected), atol=1e-6, rtol=0.0):
            raise ValueError("Lope symmetry must fix the standing pose")
        sensor = base.scene["height_scanner"]
        starts = tensor(sensor.ray_starts)
        if starts.ndim == 3:
            starts = starts[0]
        if len(starts) != CRITIC_DIM - CRITIC_PROPRIO_DIM:
            raise ValueError("Lope critic height scan width changed")
        points = [tuple(float(part) for part in row) for row in starts.detach().cpu().tolist()]
        self.height_perm = reflected_ray_indices(points)

    @staticmethod
    def reverse_joints(value):
        if value.shape[-1] != 12:
            raise ValueError("Lope requires 12 position actions")
        order = torch.tensor(JOINT_PERM, device=value.device)
        return value.index_select(-1, order) * value.new_tensor(JOINT_SIGN)

    @staticmethod
    def permute_joints(value):
        if value.shape[-1] != 12:
            raise ValueError("Lope requires 12 action standard deviations")
        return value.index_select(-1, torch.tensor(JOINT_PERM, device=value.device))

    def reverse_frame(self, value):
        if value.shape[-1] != FRAME_DIM:
            raise ValueError("Lope actor frame must contain 45 values")
        return torch.cat(
            (
                value[..., 0:3] * value.new_tensor([-1.0, 1.0, -1.0]),
                value[..., 3:6] * value.new_tensor([1.0, -1.0, 1.0]),
                value[..., 6:9] * value.new_tensor([1.0, -1.0, -1.0]),
                self.reverse_joints(value[..., 9:21]),
                self.reverse_joints(value[..., 21:33]),
                self.reverse_joints(value[..., 33:45]),
            ),
            dim=-1,
        )

    def reverse_critic(self, value):
        if value.shape[-1] != CRITIC_DIM:
            raise ValueError(f"Lope critic must contain {CRITIC_DIM} values")
        rays = torch.tensor(self.height_perm, device=value.device)
        feet = torch.tensor(FOOT_PERM, device=value.device)
        foot_start = CRITIC_PROPRIO_DIM - FOOT_FORCE_DIM
        foot_groups = tuple(
            value[..., foot_start + 4 * group : foot_start + 4 * (group + 1)].index_select(-1, feet)
            for group in range(FOOT_FORCE_DIM // 4)
        )
        return torch.cat(
            (
                value[..., 0:3] * value.new_tensor([1.0, -1.0, 1.0]),
                value[..., 3:6] * value.new_tensor([-1.0, 1.0, -1.0]),
                value[..., 6:9] * value.new_tensor([1.0, -1.0, 1.0]),
                value[..., 9:12] * value.new_tensor([1.0, -1.0, -1.0]),
                self.reverse_joints(value[..., 12:24]),
                self.reverse_joints(value[..., 24:36]),
                self.reverse_joints(value[..., 36:48]),
                self.reverse_joints(value[..., 48:60]),
                self.reverse_joints(value[..., 60:72]),
                *foot_groups,
                value[..., CRITIC_PROPRIO_DIM:].index_select(-1, rays),
            ),
            dim=-1,
        )

    def reverse_obs(self, obs):
        result = obs.clone()
        for key, value in obs.items():
            if key == "policy":
                if value.shape[-1] != POLICY_DIM:
                    raise ValueError("Lope policy history must be 10 frame-major frames")
                frames = value.reshape(*value.shape[:-1], HISTORY_LENGTH, FRAME_DIM)
                result[key] = self.reverse_frame(frames).reshape_as(value)
            elif key == "single_obs":
                result[key] = self.reverse_frame(value)
            elif key == "critic":
                result[key] = self.reverse_critic(value)
            else:
                raise ValueError("Unknown Lope observation group: " + key)
        return result

    def data_augmentation(self, obs, actions):
        return torch.cat([obs, self.reverse_obs(obs)], dim=0), torch.cat(
            [actions, self.reverse_joints(actions)], dim=0
        )


class LopeSymmetry(Go2Symmetry):
    """Reuse upstream batch ordering. Never use Go2's hip-only joint signs."""

    def __init__(self, env):
        self.mapper = LopeSymmetryMapper(env)
        self.num_aug = 2
