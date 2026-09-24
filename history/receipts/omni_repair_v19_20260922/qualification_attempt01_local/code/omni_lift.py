"""State-driven flat-ground wheel lift and landing events for omni motion.

Only measured command tracking, wheel-link height and filtered ground contact
participate.  There is no clock phase and no actor observation or action change.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


PROFILE = "omni_lift_v19"
SIDES = ("fr", "fl", "rr", "rl")
WHEEL_RADIUS_M = 0.093
CONTACT_FORCE_N = 5.0
LIFT_EVENT_WEIGHT = 0.15
LANDING_EVENT_WEIGHT = 0.25


@dataclass(frozen=True)
class LiftRules:
    min_clearance_m: float = 0.03
    ideal_clearance_m: float = 0.045
    max_clearance_m: float = 0.06
    min_air_s: float = 0.08
    max_air_s: float = 0.45
    lateral_command_min: float = 0.20
    yaw_command_min: float = 0.30
    pure_yaw_translation_max: float = 0.15
    tracking_fraction_min: float = 0.35
    lateral_error_floor: float = 0.25
    yaw_error_floor: float = 0.30

    def __post_init__(self):
        if not 0 < self.min_clearance_m < self.ideal_clearance_m < self.max_clearance_m:
            raise ValueError("clearance thresholds must be strictly increasing")
        if not 0 < self.min_air_s < self.max_air_s:
            raise ValueError("air-time thresholds must be strictly increasing")


RULES = LiftRules()


def command_tracking(
    commands: torch.Tensor, actual_velocity: torch.Tensor, rules: LiftRules = RULES
) -> dict[str, torch.Tensor]:
    """Return exact eligibility and tracking gates for lateral or pure-yaw work."""
    if commands.shape != actual_velocity.shape or commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError("commands and actual_velocity must both have shape (N, 3)")
    lateral = commands[:, 1].abs() >= rules.lateral_command_min
    pure_yaw = (
        (commands[:, :2].norm(dim=-1) <= rules.pure_yaw_translation_max)
        & (commands[:, 2].abs() >= rules.yaw_command_min)
    )

    lateral_error = (actual_velocity[:, 1] - commands[:, 1]).abs()
    yaw_error = (actual_velocity[:, 2] - commands[:, 2]).abs()
    lateral_tracking = (
        actual_velocity[:, 1] * commands[:, 1] > 0
    ) & (
        actual_velocity[:, 1].abs() >= rules.tracking_fraction_min * commands[:, 1].abs()
    ) & (
        lateral_error <= torch.maximum(
            torch.full_like(lateral_error, rules.lateral_error_floor),
            0.75 * commands[:, 1].abs(),
        )
    )
    yaw_tracking = (
        actual_velocity[:, 2] * commands[:, 2] > 0
    ) & (
        actual_velocity[:, 2].abs() >= rules.tracking_fraction_min * commands[:, 2].abs()
    ) & (
        yaw_error <= torch.maximum(
            torch.full_like(yaw_error, rules.yaw_error_floor),
            0.75 * commands[:, 2].abs(),
        )
    )
    relevant = lateral | pure_yaw
    tracking = (lateral & lateral_tracking) | (pure_yaw & yaw_tracking)
    return {
        "lateral": lateral,
        "pure_yaw": pure_yaw,
        "relevant": relevant,
        "tracking": tracking,
        "lateral_error": lateral_error,
        "yaw_error": yaw_error,
    }


def lift_quality(peak_clearance: torch.Tensor, rules: LiftRules = RULES) -> torch.Tensor:
    """Triangular quality: zero at 3/6 cm and one at 4.5 cm."""
    rising = (peak_clearance - rules.min_clearance_m) / (
        rules.ideal_clearance_m - rules.min_clearance_m
    )
    falling = (rules.max_clearance_m - peak_clearance) / (
        rules.max_clearance_m - rules.ideal_clearance_m
    )
    return torch.minimum(rising, falling).clamp(0.0, 1.0)


class LiftState:
    """Per-environment flight state; landing transitions emit one event only."""

    def __init__(self, count: int, device: str | torch.device = "cpu"):
        shape = (count, 4)
        self.previous_contact = torch.ones(shape, dtype=torch.bool, device=device)
        self.airborne = torch.zeros(shape, dtype=torch.bool, device=device)
        self.valid = torch.zeros(shape, dtype=torch.bool, device=device)
        self.air_time = torch.zeros(shape, device=device)
        self.peak_clearance = torch.zeros(shape, device=device)
        self.landing_count = torch.zeros(shape, dtype=torch.long, device=device)
        # Diagnostic accumulator only; unlike flight state it intentionally
        # survives episode resets so sparse-event incidence is observable.
        self.total_events = torch.zeros((), dtype=torch.long, device=device)

    def reset(self, reset: torch.Tensor, contacts: torch.Tensor | None = None) -> None:
        if reset.ndim != 1 or reset.shape[0] != self.airborne.shape[0]:
            raise ValueError("reset must have shape (N,)")
        self.airborne[reset] = False
        self.valid[reset] = False
        self.air_time[reset] = 0.0
        self.peak_clearance[reset] = 0.0
        self.landing_count[reset] = 0
        if contacts is not None:
            self.previous_contact[reset] = contacts[reset]

    def update(
        self,
        contacts: torch.Tensor,
        clearance: torch.Tensor,
        flat: torch.Tensor,
        commands: torch.Tensor,
        actual_velocity: torch.Tensor,
        reset: torch.Tensor,
        dt: float,
        rules: LiftRules = RULES,
    ) -> dict[str, torch.Tensor]:
        if contacts.shape != self.airborne.shape or clearance.shape != contacts.shape:
            raise ValueError("contacts and clearance must have shape (N, 4)")
        if dt <= 0:
            raise ValueError("dt must be positive")
        contacts = contacts.bool()
        reset = reset.bool()
        self.reset(reset, contacts)
        gates = command_tracking(commands, actual_velocity, rules)
        support_count = contacts.sum(dim=-1)
        allowed_env = flat.bool() & gates["relevant"] & gates["tracking"] & ~reset
        supported = support_count >= 2

        takeoff = self.previous_contact & ~contacts & ~reset[:, None]
        self.airborne |= takeoff
        self.valid = torch.where(
            takeoff,
            (allowed_env & supported)[:, None],
            self.valid,
        )

        flying = self.airborne & ~contacts
        valid_now = (allowed_env & supported)[:, None]
        self.valid &= ~flying | valid_now
        self.air_time = torch.where(flying, self.air_time + dt, self.air_time)
        self.peak_clearance = torch.where(
            flying, torch.maximum(self.peak_clearance, clearance), self.peak_clearance
        )

        landing = self.airborne & ~self.previous_contact & contacts & ~reset[:, None]
        duration_ok = (self.air_time >= rules.min_air_s) & (self.air_time <= rules.max_air_s)
        height_ok = (
            (self.peak_clearance >= rules.min_clearance_m)
            & (self.peak_clearance <= rules.max_clearance_m)
        )
        event = landing & self.valid & duration_ok & height_ok & allowed_env[:, None]
        quality = lift_quality(self.peak_clearance, rules) * event
        self.landing_count += event.long()
        self.total_events += event.sum()

        self.airborne[landing] = False
        self.valid[landing] = False
        self.air_time[landing] = 0.0
        self.peak_clearance[landing] = 0.0
        self.previous_contact.copy_(contacts)
        return {
            **gates,
            "support_count": support_count,
            "event": event,
            "lift_reward": quality.sum(dim=-1),
            "landing_reward": event.sum(dim=-1).to(clearance.dtype),
            "airborne_count": self.airborne.sum(dim=-1),
        }


def _measure(env) -> dict[str, torch.Tensor]:
    if getattr(env, "_omni_lift_step", -1) == env.common_step_counter:
        return env._omni_lift_measurement
    command = env.command_manager.get_term("base_velocity")
    robot = env.scene["robot"]
    if not hasattr(env, "_omni_lift_state"):
        ids, _ = robot.find_bodies(
            [f"{side}_foot_Link" for side in SIDES], preserve_order=True
        )
        if len(ids) != 4:
            raise ValueError("omni lift requires exactly four ordered wheel bodies")
        env._omni_lift_wheel_ids = ids
        env._omni_lift_state = LiftState(env.num_envs, env.device)
    prefix = getattr(env.cfg, "omni_lift_sensor_prefix", "omni_lift_contact")
    forces = torch.stack(
        [env.scene[f"{prefix}_{side}"].data.force_matrix_w[:, 0, 0, 2] for side in SIDES],
        dim=1,
    )
    contacts = forces > CONTACT_FORCE_N
    wheel_z = robot.data.body_link_pos_w[:, env._omni_lift_wheel_ids, 2]
    clearance = wheel_z - env.scene.env_origins[:, 2:3] - WHEEL_RADIUS_M
    flat = command.terrain_idxs == command.terrain_type2idx["flat"]
    actual = torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), dim=1)
    measured = {
        "flat ground-contact force": forces,
        "flat wheel clearance": clearance,
        "base velocity command": command.command,
        "actual body velocity": actual,
    }
    invalid = [name for name, value in measured.items() if not bool(torch.isfinite(value).all())]
    if invalid:
        raise FloatingPointError(f"nonfinite omni-lift measurement: {', '.join(invalid)}")
    result = env._omni_lift_state.update(
        contacts=contacts,
        clearance=clearance,
        flat=flat,
        commands=command.command,
        actual_velocity=actual,
        reset=env.episode_length_buf <= 1,
        dt=env.step_dt,
    )
    result.update(flat=flat, contacts=contacts, clearance=clearance, actual_velocity=actual)
    env._omni_lift_step = env.common_step_counter
    env._omni_lift_measurement = result
    return result


def reward(env, kind: str) -> torch.Tensor:
    if kind not in ("lift", "landing"):
        raise ValueError(kind)
    return _measure(env)[f"{kind}_reward"]


def configure(cfg) -> None:
    """Install two small event rewards and reuse existing real ground sensors."""
    from isaaclab.managers import RewardTermCfg
    if getattr(cfg, "mixed_profile", None) != "mixed9_v1":
        raise ValueError("omni lift requires exact mixed-terrain membership")
    existing = [hasattr(cfg.scene, f"pit_contact_{side}") for side in SIDES]
    if not all(existing):
        raise ValueError("V19 requires all four mixed9 pit_contact ground sensors")
    cfg.omni_lift_sensor_prefix = "pit_contact"
    cfg.rewards.omni_wheel_lift = RewardTermCfg(
        func=reward, weight=LIFT_EVENT_WEIGHT, params={"kind": "lift"}
    )
    cfg.rewards.omni_wheel_landing = RewardTermCfg(
        func=reward, weight=LANDING_EVENT_WEIGHT, params={"kind": "landing"}
    )
    cfg.omni_lift_profile = PROFILE


def snapshot(env) -> dict:
    state = _measure(env)
    machine = env._omni_lift_state
    flat = state["flat"]
    relevant = flat & state["relevant"]
    tracking = flat & state["tracking"]
    return {
        "profile": PROFILE,
        "scope": "exact flat terrain membership; current training environments, not held-out performance",
        "sensor": "per-wheel link filtered contact against /World/ground/terrain/mesh",
        "clearance_reference": "wheel link z - flat env origin z - 0.093 m radius",
        "weights": {"lift_event": LIFT_EVENT_WEIGHT, "landing_event": LANDING_EVENT_WEIGHT},
        "rules": RULES.__dict__,
        "flat_count": int(flat.sum()),
        "relevant_command_count": int(relevant.sum()),
        "tracking_gate_count": int(tracking.sum()),
        "active_flights": int(machine.airborne.sum()),
        "valid_active_flights": int((machine.airborne & machine.valid).sum()),
        "current_episode_landing_events": int(machine.landing_count.sum()),
        "landing_events_since_env_init": int(machine.total_events),
        "mean_support_wheels": float(state["support_count"][flat].float().mean())
        if bool(flat.any())
        else None,
    }
