"""Bound leg targets before actuator delays, leaving simulator state untouched."""
from isaaclab.envs.mdp.actions import JointPositionAction

from actuator_guard import PROFILE, limits_for_profile
from guard_math import bounded_position_target


class GuardedJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._lower, self._upper = limits[..., 0], limits[..., 1]
        self._previous = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        limits = limits_for_profile(getattr(env.cfg, "actuator_profile", PROFILE))
        self._max_delta = limits["leg_target_rate_rad_s"] * env.step_dt

    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions = bounded_position_target(
            self._processed_actions, self._previous, self._lower, self._upper, self._max_delta)
        self._previous.copy_(self._processed_actions)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # The reset event has already written the new joint pose at this point.
        self._previous[ids] = self._asset.data.joint_pos[:, self._joint_ids][ids].clamp(
            min=self._lower[ids], max=self._upper[ids])
