"""Conservative RS04/RS02 training budget and physics-step load accounting.

These are experiment budgets, not a certification of on-robot thermal capacity.
No measured joint position or velocity is overwritten by this module.
"""
import torch

from guard_math import guard_costs


PROFILE = "rs0402_guard_v1"
LIMITS = {
    "leg_torque_nm": 35.0,
    "wheel_torque_nm": 6.0,
    "leg_speed_rad_s": 15.0,
    "wheel_speed_rad_s": 30.0,
    "leg_target_rate_rad_s": 6.0,
    "speed_penalty_fraction": 0.8,
    "speed_termination_fraction": 1.02,
    "hard_position_termination_rad": 0.01,
}
PEAK_PROFILE = "rs0402_peak_v2"
PEAK_LIMITS = {
    **LIMITS,
    "leg_torque_nm": 120.0,
    "wheel_torque_nm": 17.0,
    "leg_load_reference_nm": 35.0,
    "wheel_load_reference_nm": 6.0,
}
SLEW_PROFILE = "rs0402_slew12_v3"
SLEW_LIMITS = {**PEAK_LIMITS, "leg_target_rate_rad_s": 12.0}


def limits_for_profile(profile):
    if profile == PROFILE:
        return dict(LIMITS)
    if profile == PEAK_PROFILE:
        return dict(PEAK_LIMITS)
    if profile == SLEW_PROFILE:
        return dict(SLEW_LIMITS)
    raise ValueError(f"Unknown actuator profile: {profile}")


def configure_guard(cfg, profile):
    if profile == "original":
        return
    limits = limits_for_profile(profile)
    cfg.actuator_profile = profile
    from isaaclab.managers import RewardTermCfg, TerminationTermCfg
    from guard_actions import GuardedJointPositionAction

    for name in ("hip", "thigh", "calf"):
        actuator = cfg.scene.robot.actuators[name]
        actuator.effort_limit = actuator.effort_limit_sim = limits["leg_torque_nm"]
        actuator.velocity_limit = actuator.velocity_limit_sim = limits["leg_speed_rad_s"]
        # Keep the original stall-curve parameter; each profile's effort cap
        # applies in both directions. This is not a measured RS04 motor curve.
    wheel = cfg.scene.robot.actuators["wheel"]
    wheel.effort_limit = wheel.effort_limit_sim = limits["wheel_torque_nm"]
    wheel.velocity_limit = wheel.velocity_limit_sim = limits["wheel_speed_rad_s"]
    cfg.actions.joint_vel.clip = {".*": (-limits["wheel_speed_rad_s"], limits["wheel_speed_rad_s"])}
    cfg.actions.joint_pos.class_type = GuardedJointPositionAction
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    # All four physics steps contribute, rather than only the last substep.
    cfg.rewards.joint_torques_l2.func = measured_cost
    cfg.rewards.joint_torques_l2.params = {"key": "torque_square"}
    cfg.rewards.joint_acc_l2.func = measured_cost
    cfg.rewards.joint_acc_l2.params = {"key": "acceleration_square"}
    for name, key, weight in (
        ("actuator_load", "torque_mean_square", -0.5),
        ("actuator_speed", "speed_excess", -5.0),
        ("actuator_saturation", "saturation", -0.05),
    ):
        setattr(cfg.rewards, name, RewardTermCfg(func=measured_cost, weight=weight, params={"key": key}))
    cfg.terminations.actuator_violation = TerminationTermCfg(func=actuator_violation)
    cfg.rewards.actuator_violation = RewardTermCfg(func=violation_cost, weight=-100.0)


def measured_cost(env, key):
    return env.actuator_guard.costs[key] / env.cfg.decimation


def actuator_violation(env):
    return env.actuator_guard.violation


def violation_cost(env):
    return env.actuator_guard.violation.float()


class ActuatorGuard:
    """Attach after environment construction, before wrapper reset or stepping."""

    def __init__(self, env, profile=None):
        self.env = env
        self.profile = profile or getattr(env.cfg, "actuator_profile", PROFILE)
        self.limits = limits_for_profile(self.profile)
        self.robot = env.scene["robot"]
        self.names = self.robot.joint_names
        self.leg = torch.tensor([not n.endswith("foot_joint") for n in self.names], device=env.device)
        self.torque_limit = torch.where(self.leg, self.limits["leg_torque_nm"], self.limits["wheel_torque_nm"])
        self.speed_limit = torch.where(self.leg, self.limits["leg_speed_rad_s"], self.limits["wheel_speed_rad_s"])
        self.load_reference = torch.where(
            self.leg, self.limits.get("leg_load_reference_nm", self.limits["leg_torque_nm"]),
            self.limits.get("wheel_load_reference_nm", self.limits["wheel_torque_nm"]))
        self.hard = self.robot.root_physx_view.get_dof_limits().to(env.device)
        torch.testing.assert_close(self.robot.root_physx_view.get_dof_max_forces().to(env.device),
                                   self.torque_limit.expand(env.num_envs, -1))
        torch.testing.assert_close(self.robot.root_physx_view.get_dof_max_velocities().to(env.device),
                                   self.speed_limit.expand(env.num_envs, -1))
        self.costs = {key: torch.zeros(env.num_envs, device=env.device) for key in (
            "torque_mean_square", "speed_excess", "saturation", "torque_square", "acceleration_square")}
        self.violation = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.streak = torch.zeros((env.num_envs, len(self.names)), device=env.device)
        self._last_sim_step = env._sim_step_counter
        self.reset_statistics()
        env.actuator_guard = self
        original_process = env.action_manager.process_action
        original_update = env.scene.update

        def process(action):
            for value in self.costs.values():
                value.zero_()
            self.violation.zero_()
            self.streak[env.episode_length_buf == 0] = 0
            return original_process(action)

        def update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            if env._sim_step_counter != self._last_sim_step:
                self._last_sim_step = env._sim_step_counter
                self.measure()
            return result

        env.action_manager.process_action = process
        env.scene.update = update

    def reset_statistics(self):
        self.samples_per_joint = 0
        self.statistics = {key: torch.zeros(len(self.names), device=self.env.device) for key in (
            "torque_sq_sum", "peak_torque", "peak_speed", "max_hard_position_excess",
            "max_target_position_excess", "clipped_samples", "overspeed_samples",
            "torque_violation_samples", "max_clipped_duration_s")}
        # Do not clear the per-environment streak at a logging boundary.

    def measure(self):
        data = self.robot.data
        torque, computed, velocity = data.applied_torque, data.computed_torque, data.joint_vel
        for key, value in guard_costs(torque, computed, velocity, self.torque_limit, self.speed_limit,
                                      self.load_reference).items():
            self.costs[key].add_(value)
        self.costs["torque_square"].add_(torque.square().sum(-1))
        self.costs["acceleration_square"].add_(data.joint_acc.square().sum(-1))
        pos_excess = torch.maximum(self.hard[..., 0] - data.joint_pos,
                                   data.joint_pos - self.hard[..., 1]).clamp_min(0) * self.leg
        target_excess = torch.maximum(self.hard[..., 0] - data.joint_pos_target,
                                      data.joint_pos_target - self.hard[..., 1]).clamp_min(0) * self.leg
        overspeed = velocity.abs() > self.speed_limit + 0.001
        torque_violation = torque.abs() > self.torque_limit + 0.001
        self.violation.logical_or_(
            (velocity.abs() > self.speed_limit * self.limits["speed_termination_fraction"]).any(-1)
            | (pos_excess > self.limits["hard_position_termination_rad"]).any(-1)
            | torque_violation.any(-1)
        )
        clipped = (computed - torque).abs() > 0.001
        self.streak = torch.where(clipped, self.streak + self.env.physics_dt, 0.0)
        self.samples_per_joint += self.env.num_envs
        self.statistics["torque_sq_sum"].add_(torque.square().sum(0))
        for key, value in (
            ("peak_torque", torque.abs()), ("peak_speed", velocity.abs()),
            ("max_hard_position_excess", pos_excess), ("max_target_position_excess", target_excess),
            ("max_clipped_duration_s", self.streak),
        ):
            self.statistics[key] = torch.maximum(self.statistics[key], value.amax(0))
        for key, value in (("clipped_samples", clipped), ("overspeed_samples", overspeed),
                           ("torque_violation_samples", torque_violation)):
            self.statistics[key].add_(value.sum(0))

    def snapshot(self):
        values = {key: value.detach().cpu().tolist() for key, value in self.statistics.items()}
        denom = max(1, self.samples_per_joint)
        joints = {name: {key: value[i] for key, value in values.items() if key != "torque_sq_sum"}
                  for i, name in enumerate(self.names)}
        for i, name in enumerate(self.names):
            joints[name]["torque_rms_nm"] = (values["torque_sq_sum"][i] / denom) ** 0.5
            joints[name]["clipped_sample_fraction"] = values["clipped_samples"][i] / denom
        return {"profile": self.profile, "limits": self.limits, "samples_per_joint": self.samples_per_joint,
                "joints": joints, "scope": "all training environments since last log; 200 Hz, including resets"}
