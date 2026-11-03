"""Default-pose configuration and evidence; no additional hip target guard.

The module name is retained for the existing receipt/export integration. V23 uses
the original GuardedJointPositionAction unchanged. A pose departure from 0.12 is
not a limit violation; the motion envelope remains the inherited +/-0.369 rad.
"""
import copy

import torch

from guard_actions import GuardedJointPositionAction
from hip_contract import DEFAULT_INWARD, INWARD_SIGNS, PROFILE, SOFT_LIMIT, contract, joint_contract


def configure(cfg):
    if cfg.actions.joint_pos.class_type is not GuardedJointPositionAction:
        raise ValueError("V23 requires the inherited position/slew guard")
    cfg.scene.robot.init_state = copy.deepcopy(cfg.scene.robot.init_state)
    for name in INWARD_SIGNS:
        cfg.scene.robot.init_state.joint_pos[name] = joint_contract(name)["default_rad"]
    cfg.hip_guard_profile = PROFILE


def validate_offset(default, actual, cached_default, offset_range):
    """Keep the inherited reset motor-zero randomization around the new nominal."""
    if cached_default is None:
        torch.testing.assert_close(actual, default, rtol=0, atol=1e-6)
        return
    torch.testing.assert_close(cached_default, default, rtol=0, atol=1e-6)
    noise = actual - cached_default
    if not bool(((noise >= offset_range[0]-1e-6) & (noise <= offset_range[1]+1e-6)).all()):
        raise ValueError("Action zero-offset noise exceeds inherited bounds")


def validate_runtime(env):
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    if type(term) is not GuardedJointPositionAction:
        raise TypeError("V23 must not install an additional action guard")
    physical = robot.root_physx_view.get_dof_limits()
    bounds = robot.data.soft_joint_pos_limits[:, term.joint_indices]
    torch.testing.assert_close(term._lower, bounds[..., 0], rtol=0, atol=0)
    torch.testing.assert_close(term._upper, bounds[..., 1], rtol=0, atol=0)
    for name in INWARD_SIGNS:
        j = robot.joint_names.index(name)
        expected = joint_contract(name)
        torch.testing.assert_close(physical[:, j], torch.tensor([-.41, .41]).expand_as(physical[:, j]),
                                   rtol=0, atol=1e-6)
        for axis, key in ((0, "lower_rad"), (1, "upper_rad")):
            torch.testing.assert_close(robot.data.soft_joint_pos_limits[:, j, axis],
                                       torch.full_like(robot.data.joint_pos[:, j], expected[key]))
        torch.testing.assert_close(robot.data.default_joint_pos[:, j],
                                   torch.full_like(robot.data.joint_pos[:, j], expected["default_rad"]))
        policy_j = list(term.joint_indices).index(j)
        cached = getattr(env, '_default_action_offset_joint_pos', None)
        cached_joint = cached[:, policy_j] if cached is not None else None
        bias_event = env.cfg.events.randomize_motor_zero_offset
        if bias_event is None or tuple(bias_event.params['offset_range']) != (-.035, .035):
            raise ValueError("Inherited motor zero-offset randomization changed")
        validate_offset(robot.data.default_joint_pos[:, j], term._offset[:, policy_j],
                        cached_joint, bias_event.params['offset_range'])
    if env.reward_manager.get_term_cfg("joint_pos_limits").weight != -2.:
        raise ValueError("Inherited soft-limit penalty unexpectedly changed")
    if abs(term._max_delta - .24) > 1e-6:
        raise ValueError("Inherited leg target slew unexpectedly changed")
    return contract()


class HipEvidence:
    """Observe inward motion without treating the default pose as a hard bound."""
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.term = env.action_manager.get_term("joint_pos")
        self.names = list(INWARD_SIGNS)
        self.ids = [self.robot.joint_names.index(n) for n in self.names]
        self.action_ids = [list(self.term.joint_indices).index(i) for i in self.ids]
        self.signs = torch.tensor(list(INWARD_SIGNS.values()), device=env.device)
        self.last_step = -1
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
        self.above_default = torch.zeros(4, device=self.env.device)
        self.pose_abs_sum = torch.zeros_like(self.above_default)
        self.target_abs_peak = torch.zeros_like(self.above_default)

    def measure(self):
        actual = self.robot.data.joint_pos[:, self.ids] * self.signs
        values = dict(actual=actual,
                      target=self.term.processed_actions[:, self.action_ids] * self.signs,
                      requested=self.term.requested_actions[:, self.action_ids] * self.signs)
        for key, value in values.items():
            self.peak[key] = torch.maximum(self.peak[key], value.amax(0))
        self.above_default += (actual > DEFAULT_INWARD + .001).sum(0)
        self.pose_abs_sum += (actual - DEFAULT_INWARD).abs().sum(0)
        self.target_abs_peak = torch.maximum(self.target_abs_peak, values['target'].abs().amax(0))
        self.samples += self.env.num_envs

    def snapshot(self):
        if bool((self.target_abs_peak > SOFT_LIMIT + 1e-6).any()):
            raise RuntimeError("Hip target escaped the inherited soft limits")
        return dict(profile=PROFILE, samples_per_hip=self.samples,
                    peak_inward_rad={k: v.tolist() for k, v in self.peak.items()} if self.samples else None,
                    actual_above_default_fraction=(self.above_default / max(1, self.samples)).tolist(),
                    actual_default_deviation_mae_rad=(self.pose_abs_sum / max(1, self.samples)).tolist(),
                    target_abs_peak_rad=self.target_abs_peak.tolist(), joint_names=self.names,
                    default_is_not_a_limit=True,
                    scope="200Hz all environments including resets; pose departure is not a limit violation")
