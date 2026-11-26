"""45/275 Lope observations and privileged 12-DoF tensor helpers."""
from dataclasses import dataclass

import torch
from go2_rl_lab.tasks.mixed.mdp.observations import foot_contact_force_norm
from go2_rl_lab.utils.height_scan import ray_caster_height_scan
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from lope_rl_lab.assets.lope_cfg import ACTION_DIM, FOOT_BODIES, JOINT_NAMES
from lope_rl_lab.tasks.locomotion.config import CRITIC_DIM, FEATURE_NAMES, FRAME_DIM


@dataclass(frozen=True)
class FeatureScales:
    acceleration: float = 1e-4
    torque: float = 0.01
    contact_force: float = 1e-3


def tensor(value):
    """Isaac Lab 3 ProxyArray or a Tensor. Do not copy to CPU."""
    result = value if isinstance(value, torch.Tensor) else getattr(value, "torch", None)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Expected torch Tensor or ProxyArray.torch")
    return result


def joint_indices(robot):
    return [robot.joint_names.index(name) for name in JOINT_NAMES]


def foot_indices(names):
    return [list(names).index(name) for name in FOOT_BODIES]


def _joint_ids(env):
    if not hasattr(env, "_lope_joint_order"):
        env._lope_joint_order = joint_indices(env.scene["robot"])
    return env._lope_joint_order


def _command(env):
    command = env.command_manager.get_command("base_velocity")
    return tensor(command)


def policy_frame(env):
    robot = env.scene["robot"]
    ids = _joint_ids(env)
    relative_position = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:, ids]
    relative_velocity = (robot.data.joint_vel.torch - robot.data.default_joint_vel.torch)[:, ids]
    raw = (
        robot.data.root_ang_vel_b.torch,
        robot.data.projected_gravity_b.torch,
        _command(env),
        relative_position,
        relative_velocity,
        env.action_manager.action,
    )
    if env.action_manager.action.shape[-1] != ACTION_DIM:
        raise RuntimeError("Lope policy expects 12 position actions and no wheel velocity")
    command = raw[2] * raw[2].new_tensor((0.5, 1.0, 1.0))
    scaled = (raw[0], raw[1], command, raw[3], raw[4], raw[5])
    scales = (0.25, 1.0, 1.0, 1.0, 0.05, 1.0)
    return torch.cat(tuple(value.clamp(-100, 100) * scale for value, scale in zip(scaled, scales)), dim=-1)


def _raw_policy_frame(env):
    robot = env.scene["robot"]
    ids = _joint_ids(env)
    return torch.cat(
        (
            robot.data.root_ang_vel_b.torch,
            robot.data.projected_gravity_b.torch,
            _command(env),
            (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:, ids],
            (robot.data.joint_vel.torch - robot.data.default_joint_vel.torch)[:, ids],
            env.action_manager.action,
        ),
        dim=-1,
    )


class _SharedFrameNoise:
    def __init__(self):
        self.sample = None
        self.valid = None
        self.step = None

    def reset(self, env_ids=None):
        if self.valid is not None:
            self.valid[slice(None) if env_ids is None else env_ids] = False

    def apply(self, raw, step):
        if raw.ndim != 2 or raw.shape[1] != FRAME_DIM:
            raise ValueError("Expected one unscaled 45-value Lope frame per environment")
        if self.sample is None:
            self.sample = torch.zeros_like(raw)
            self.valid = torch.zeros(len(raw), dtype=torch.bool, device=raw.device)
        if self.step != step:
            self.valid.zero_()
            self.step = step
        refresh = ~self.valid
        amplitude = raw.new_tensor([0.2] * 3 + [0.05] * 3 + [0.0] * 3 + [0.03] * 12 + [2.0] * 12 + [0.0] * 12)
        self.sample[refresh] = (2 * torch.rand_like(raw[refresh]) - 1) * amplitude
        self.valid[refresh] = True
        scale = raw.new_tensor([0.25] * 3 + [1.0] * 3 + [0.5, 1.0, 1.0] + [1.0] * 12 + [0.05] * 12 + [1.0] * 12)
        return (raw + self.sample).clamp(-100, 100) * scale


class NoisyPolicyFrame(ManagerTermBase):
    """One noise sample shared by the history term and the current frame."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not hasattr(env, "_lope_policy_frame_noise"):
            env._lope_policy_frame_noise = _SharedFrameNoise()
        self.state = env._lope_policy_frame_noise

    def reset(self, env_ids=None):
        self.state.reset(env_ids)

    def __call__(self, env):
        observations = env.cfg.observations
        enabled = observations.policy.enable_corruption
        if enabled != observations.single_obs.enable_corruption:
            raise ValueError("Lope policy and single_obs must use identical corruption settings")
        if not enabled:
            return policy_frame(env)
        return self.state.apply(_raw_policy_frame(env), int(env.common_step_counter))


def critic_frame(env):
    robot = env.scene["robot"]
    ids = _joint_ids(env)
    sensor = env.scene["contact_forces"]
    if not hasattr(env, "_lope_foot_order"):
        env._lope_foot_order = foot_indices(sensor.body_names)
        if [sensor.body_names[index] for index in env._lope_foot_order] != list(FOOT_BODIES):
            raise RuntimeError("Foot contact order must be FR, FL, RR, RL")
    height = ray_caster_height_scan(env, SceneEntityCfg("height_scanner"), offset=0.5).clamp(-1, 1) * 2.5
    sensor_cfg = SceneEntityCfg("contact_forces")
    sensor_cfg.body_ids = list(env._lope_foot_order)
    foot_force = foot_contact_force_norm(env, sensor_cfg) * 1e-3
    proprioception = torch.cat(
        (
            robot.data.root_lin_vel_b.torch * robot.data.root_lin_vel_b.torch.new_tensor((0.5, 1.0, 1.0)),
            robot.data.root_ang_vel_b.torch * 0.25,
            robot.data.projected_gravity_b.torch,
            _command(env) * _command(env).new_tensor((0.5, 1.0, 1.0)),
            (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:, ids],
            (robot.data.joint_vel.torch - robot.data.default_joint_vel.torch)[:, ids] * 0.05,
            env.action_manager.action,
            robot.data.joint_acc.torch[:, ids] * 1e-4,
            robot.actuators.applied_effort.torch[:, ids] * 0.01,
            foot_force,
        ),
        dim=-1,
    ).clamp(-100, 100)
    result = torch.cat((proprioception, height), dim=-1)
    if result.shape != (env.num_envs, CRITIC_DIM):
        raise RuntimeError(f"Expected Lope critic observations [N,{CRITIC_DIM}], got {tuple(result.shape)}")
    if list(JOINT_NAMES) != [robot.joint_names[index] for index in ids]:
        raise RuntimeError("Critic joint order left the canonical 12-DoF list")
    return result
