"""45/263 Lope observations. Joint velocity and action blocks are 12-wide."""
import torch
from isaaclab.managers import ManagerTermBase

from lope_lab3.spec import ACTION_DIM, CRITIC_DIM, FOOT_BODIES, FRAME_DIM, JOINT_NAMES
from lope_lab3.tasks.lope.mdp.privileged import foot_indices, joint_indices, tensor


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
    scales = (0.25, 1.0, 1.0, 1.0, 0.05, 1.0)
    return torch.cat(tuple(value.clamp(-100, 100) * scale for value, scale in zip(raw, scales)), dim=-1)


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
        scale = raw.new_tensor([0.25] * 3 + [1.0] * 18 + [0.05] * 12 + [1.0] * 12)
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
    scanner = env.scene["height_scanner"]
    height = (scanner.data.pos_w.torch[:, 2:3] - scanner.data.ray_hits_w.torch[..., 2] - 0.5).clamp(-1, 1) * 2.5
    forces = sensor.data.net_normal_forces_w.torch[:, env._lope_foot_order]
    proprioception = torch.cat(
        (
            robot.data.root_lin_vel_b.torch * 2.0,
            robot.data.root_ang_vel_b.torch * 0.25,
            robot.data.projected_gravity_b.torch,
            _command(env),
            (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:, ids],
            (robot.data.joint_vel.torch - robot.data.default_joint_vel.torch)[:, ids] * 0.05,
            env.action_manager.action,
            robot.data.joint_acc.torch[:, ids] * 1e-4,
            robot.actuators.applied_effort.torch[:, ids] * 0.01,
            torch.linalg.vector_norm(forces, dim=-1) * 1e-3,
        ),
        dim=-1,
    ).clamp(-100, 100)
    result = torch.cat((proprioception, height), dim=-1)
    if result.shape != (env.num_envs, CRITIC_DIM) or not torch.isfinite(result).all():
        raise RuntimeError(f"Expected finite Lope critic observations [N,{CRITIC_DIM}], got {tuple(result.shape)}")
    if list(JOINT_NAMES) != [robot.joint_names[index] for index in ids]:
        raise RuntimeError("Critic joint order left the canonical 12-DoF list")
    return result
