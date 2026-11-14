"""Lope velocity sampling. Ranges stay inside the historical rough-task gate.

This is not Go2's limit curriculum and it does not raise flat speed to
4.5 m/s. Lateral and yaw commands stay at the rough_cfg values.
"""
import torch
from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

from lope_lab3.spec import COMMAND_RANGES, FORWARD_SPEED, REL_STANDING_ENVS, RESAMPLE_TIME_S, advance_forward_ceiling
from lope_lab3.tasks.lope.mdp.privileged import tensor


def _ids(env_ids, num_envs, device):
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)
    ids = torch.as_tensor(env_ids, device=device)
    if ids.dtype == torch.bool:
        return ids.nonzero().flatten()
    return ids.long()


class ForwardSpeedCurriculum(ManagerTermBase):
    """Raise only the forward ceiling after successful forward episodes.

    Ported from ``SmallDogRoughEnvCfg`` / ``ForwardSpeedCurriculum``. The R6
    extension to 1.5 m/s is a later warm start and is not this gate.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.count = 0
        self.passed = 0

    def __call__(self, env, env_ids, max_speed, increment, batch_size, success_ratio, x_lower):
        command = env.command_manager.get_term("base_velocity")
        ceiling = float(command.cfg.ranges.lin_vel_x[1])
        if env.common_step_counter < env.max_episode_length:
            return ceiling
        ids = _ids(env_ids, env.num_envs, env.device)
        cmd = tensor(command.command)[ids]
        eligible = (
            (cmd[:, 0] >= 0.6 * ceiling)
            & (cmd[:, 1].abs() < 0.06)
            & (cmd[:, 2].abs() < 0.15)
            & (env.episode_length_buf[ids] > 0)
        )
        selected = ids[eligible]
        self.count += int(selected.numel())
        if selected.numel():
            tolerance = torch.clamp(tensor(command.command)[selected, 0] * 0.25, min=0.08)
            passed = env.reset_time_outs[selected] & (command.metrics["error_vel_xy"][selected] < tolerance)
            self.passed += int(passed.sum())
        new_ceiling, advanced = advance_forward_ceiling(
            ceiling,
            self.passed,
            self.count,
            {
                "max_speed": max_speed,
                "increment": increment,
                "batch_size": batch_size,
                "success_ratio": success_ratio,
                "x_lower": x_lower,
            },
        )
        if self.count >= batch_size:
            if advanced:
                command.cfg.ranges.lin_vel_x = (x_lower, new_ceiling)
            self.count = 0
            self.passed = 0
            return new_ceiling
        return ceiling


def terrain_levels_after_episode(env, env_ids):
    """Isaac Lab distance rule used by Lope's rough task, skipping the first reset.

    Promotion is half a tile. Demotion is half the commanded distance over the
    episode. This is not Thunder's travel tracker and not Go2's full-speed cap.
    """
    ids = _ids(env_ids, env.num_envs, env.device)
    ids = ids[env.episode_length_buf[ids] > 0]
    terrain = env.scene.terrain
    if not len(ids):
        return terrain.terrain_levels.float().mean()
    command = tensor(env.command_manager.get_command("base_velocity"))
    root = tensor(env.scene["robot"].data.root_pos_w)[ids, :2]
    distance = torch.linalg.vector_norm(root - env.scene.env_origins[ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.linalg.vector_norm(command[ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down = move_down & ~move_up
    terrain.update_env_origins(ids, move_up, move_down)
    return terrain.terrain_levels.float().mean()


@configclass
class LopeVelocityCommandCfg(lab_mdp.UniformVelocityCommandCfg):
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = RESAMPLE_TIME_S
    rel_standing_envs: float = REL_STANDING_ENVS
    rel_heading_envs: float = 0.0
    heading_command: bool = False
    debug_vis: bool = False
    ranges: lab_mdp.UniformVelocityCommandCfg.Ranges = lab_mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=COMMAND_RANGES["lin_vel_x"],
        lin_vel_y=COMMAND_RANGES["lin_vel_y"],
        ang_vel_z=COMMAND_RANGES["ang_vel_z"],
    )
