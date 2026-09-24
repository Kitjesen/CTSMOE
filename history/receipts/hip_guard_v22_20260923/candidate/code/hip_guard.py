"""V22 hip-only operating envelope, default pose, and physical-step evidence."""
import copy

import torch

from guard_actions import GuardedJointPositionAction
from hip_contract import INWARD_MAX, INWARD_SIGNS, PROFILE, contract, joint_contract


class HipGuardedJointPositionAction(GuardedJointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hip_guard_physical_limits_before = self._asset.root_physx_view.get_dof_limits().clone()
        # The asset soft bounds are also read by the existing joint_pos_limits
        # reward and reset sampler. Do not change PhysX mechanical hard limits.
        for name in INWARD_SIGNS:
            j = self._asset.joint_names.index(name)
            limits = joint_contract(name)
            self._asset.data.soft_joint_pos_limits[:, j, 0] = limits["lower_rad"]
            self._asset.data.soft_joint_pos_limits[:, j, 1] = limits["upper_rad"]
            torch.testing.assert_close(self._asset.data.default_joint_pos[:, j],
                                       torch.full_like(self._asset.data.default_joint_pos[:, j], limits["default_rad"]))
        # Parent uses advanced indexing: refresh its copies after installing bounds.
        bounds = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._lower, self._upper = bounds[..., 0], bounds[..., 1]
        self._previous = self._asset.data.default_joint_pos[:, self._joint_ids].clone()


def configure(cfg):
    if cfg.actions.joint_pos.class_type is not GuardedJointPositionAction:
        raise ValueError("V22 requires the inherited position/slew guard")
    cfg.scene.robot.init_state = copy.deepcopy(cfg.scene.robot.init_state)
    for name in INWARD_SIGNS:
        cfg.scene.robot.init_state.joint_pos[name] = joint_contract(name)["default_rad"]
    cfg.actions.joint_pos.class_type = HipGuardedJointPositionAction
    cfg.hip_guard_profile = PROFILE


def validate_runtime(env):
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    if not isinstance(term, HipGuardedJointPositionAction):
        raise TypeError("V22 target envelope is not active")
    physical = robot.root_physx_view.get_dof_limits()
    torch.testing.assert_close(physical, term.hip_guard_physical_limits_before, rtol=0, atol=0)
    for name in INWARD_SIGNS:
        j = robot.joint_names.index(name)
        bounds = joint_contract(name)
        torch.testing.assert_close(physical[:, j], torch.tensor([-.41, .41]).expand_as(physical[:, j]),
                                   rtol=0, atol=1e-6)
        for axis, key in ((0, "lower_rad"), (1, "upper_rad")):
            torch.testing.assert_close(robot.data.soft_joint_pos_limits[:, j, axis],
                                       torch.full_like(robot.data.joint_pos[:, j], bounds[key]))
        torch.testing.assert_close(robot.data.default_joint_pos[:, j],
                                   torch.full_like(robot.data.joint_pos[:, j], bounds["default_rad"]))
        policy_j = list(term.joint_indices).index(j)
        torch.testing.assert_close(term._offset[:, policy_j], robot.data.default_joint_pos[:, j])
    if env.reward_manager.get_term_cfg("joint_pos_limits").weight != -2.:
        raise ValueError("Inherited soft-limit penalty unexpectedly changed")
    if abs(term._max_delta - .24) > 1e-6:
        raise ValueError("Inherited leg target slew unexpectedly changed")
    return contract()


class HipEvidence:
    """Record actual overshoot instead of hiding it by clamping simulator state."""
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.term = env.action_manager.get_term("joint_pos")
        self.names = list(INWARD_SIGNS)
        self.ids = [self.robot.joint_names.index(n) for n in self.names]
        self.action_ids = [list(self.term.joint_indices).index(i) for i in self.ids]
        self.signs = torch.tensor(list(INWARD_SIGNS.values()), device=env.device)
        self.last_step = -1
        self.streak = torch.zeros(env.num_envs, 4, device=env.device)
        self.reset_statistics()
        original_update = env.scene.update

        def update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            if env._sim_step_counter != self.last_step:
                self.last_step = env._sim_step_counter
                self.measure()
            return result

        env.scene.update = update

    def reset_statistics(self):
        self.samples = 0
        self.peak = {k: torch.full((4,), -torch.inf, device=self.env.device)
                     for k in ("actual", "target", "requested")}
        self.excess = torch.zeros(4, device=self.env.device)
        self.requested_excess = torch.zeros_like(self.excess)
        self.longest = torch.zeros_like(self.excess)

    def measure(self):
        actual = self.robot.data.joint_pos[:, self.ids] * self.signs
        values = dict(actual=actual,
                      target=self.term.processed_actions[:, self.action_ids] * self.signs,
                      requested=self.term.requested_actions[:, self.action_ids] * self.signs)
        for key, value in values.items():
            self.peak[key] = torch.maximum(self.peak[key], value.amax(0))
        over = actual > INWARD_MAX + .001
        self.excess += over.sum(0)
        self.requested_excess += (values["requested"] > INWARD_MAX + .001).sum(0)
        self.streak[self.env.episode_length_buf == 0] = 0
        self.streak = torch.where(over, self.streak + self.env.physics_dt, 0.)
        self.longest = torch.maximum(self.longest, self.streak.amax(0))
        self.samples += self.env.num_envs

    def snapshot(self):
        data = {k: v.tolist() for k, v in self.peak.items()} if self.samples else None
        if data and max(data["target"]) > INWARD_MAX + 1e-6:
            raise RuntimeError("Hip processed target escaped inward envelope")
        return dict(profile=PROFILE, samples_per_hip=self.samples,
                    peak_inward_rad=data,
                    actual_above_limit_fraction=(self.excess / max(1, self.samples)).tolist(),
                    requested_above_limit_fraction=(self.requested_excess / max(1, self.samples)).tolist(),
                    longest_actual_excess_s=self.longest.tolist(), joint_names=self.names,
                    scope="200Hz all environments, actual overshoot retained; not hardware certification")
