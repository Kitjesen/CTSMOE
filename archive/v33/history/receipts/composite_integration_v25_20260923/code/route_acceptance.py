"""Whole-body terminal platform acceptance for the composite route.

This is deliberately separate from V23's 4 m radius and stair-axis timeout.
Its inputs must be observations from the current simulation control step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from composite_route import RouteSpec


@dataclass(frozen=True)
class AcceptanceSpec:
    edge_margin_m: float = 0.05
    border_margin_m: float = 0.05
    dwell_required_s: float = 2.0
    max_translation_mps: float = 0.15
    max_yaw_rps: float = 0.30
    max_tilt_rad: float = 0.30

    def __post_init__(self) -> None:
        values = (
            self.edge_margin_m, self.border_margin_m, self.dwell_required_s,
            self.max_translation_mps, self.max_yaw_rps, self.max_tilt_rad,
        )
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Acceptance limits must be nonnegative and finite")
        if self.dwell_required_s == 0:
            raise ValueError("A terminal support dwell is required")


def route_section(x_m: float, route: RouteSpec) -> str:
    """Human-readable route phase from tile-local X, not distance from spawn."""
    if x_m < route.approach_end_x_m:
        return "approach"
    if x_m < route.slope_end_x_m:
        return "slope"
    if x_m < route.transition_end_x_m:
        return "transition_platform"
    if x_m < route.final_start_x_m:
        return "stairs"
    return "final_platform"


class RouteJudge:
    """Stateful, per-episode evaluator; use one instance per environment."""

    def __init__(self, route: RouteSpec, acceptance: AcceptanceSpec | None = None):
        self.route = route
        self.acceptance = acceptance or AcceptanceSpec()
        self.reset()

    def reset(self) -> None:
        self.dwell_s = 0.0
        self.furthest_whole_body_x_m = -math.inf
        self.success = False
        self.physical_failure = False
        self.seen_slope_support = False
        self.seen_stair_support = False

    def update(
        self,
        *,
        dt_s: float,
        wheel_xy,
        body_hull_xy,
        wheel_support,
        translation_speed_mps: float,
        yaw_rate_rps: float,
        body_tilt_rad: float,
        physical_failure: bool = False,
    ) -> dict:
        wheels = np.asarray(wheel_xy, dtype=np.float64)
        body = np.asarray(body_hull_xy, dtype=np.float64)
        support = np.asarray(wheel_support, dtype=bool)
        if (wheels.shape != (4, 2) or body.ndim != 2 or body.shape[1] != 2
                or len(body) < 4 or support.shape != (4,)
                or not np.isfinite(wheels).all() or not np.isfinite(body).all()):
            raise ValueError("Expected four finite wheel XY, body hull XY, and four support flags")
        if not (math.isfinite(dt_s) and dt_s > 0):
            raise ValueError("Control-step duration must be positive and finite")
        motion = (translation_speed_mps, yaw_rate_rps, body_tilt_rad)
        if not all(math.isfinite(value) for value in motion):
            raise ValueError("Motion measurements must be finite")

        self.physical_failure |= bool(physical_failure)
        points = np.vstack((wheels, body))
        rear_x = float(points[:, 0].min())
        self.furthest_whole_body_x_m = max(self.furthest_whole_body_x_m, rear_x)
        supported_x = wheels[support, 0]
        self.seen_slope_support |= bool(((supported_x >= self.route.approach_end_x_m)
                                         & (supported_x < self.route.slope_end_x_m)).any())
        self.seen_stair_support |= bool(((supported_x >= self.route.transition_end_x_m)
                                         & (supported_x < self.route.final_start_x_m)).any())
        lower_x = self.route.final_start_x_m + self.acceptance.edge_margin_m
        upper_x = self.route.size_x_m - self.route.tile_border_m - self.acceptance.border_margin_m
        lower_y = self.route.tile_border_m + self.acceptance.border_margin_m
        upper_y = self.route.size_y_m - self.route.tile_border_m - self.acceptance.border_margin_m
        completely_on_platform = bool(
            (points[:, 0] >= lower_x).all() and (points[:, 0] <= upper_x).all()
            and (points[:, 1] >= lower_y).all() and (points[:, 1] <= upper_y).all()
        )
        stable = bool(
            completely_on_platform and support.all()
            and self.seen_slope_support and self.seen_stair_support
            and abs(translation_speed_mps) <= self.acceptance.max_translation_mps
            and abs(yaw_rate_rps) <= self.acceptance.max_yaw_rps
            and abs(body_tilt_rad) <= self.acceptance.max_tilt_rad
            and not self.physical_failure
        )
        if not self.success:
            self.dwell_s = self.dwell_s + dt_s if stable else 0.0
            self.success = self.dwell_s + 1e-9 >= self.acceptance.dwell_required_s
        return {
            "all_wheels_and_body_on_final_platform": completely_on_platform,
            "stable_support_now": stable,
            "continuous_support_s": self.dwell_s,
            "furthest_whole_body_x_m": self.furthest_whole_body_x_m,
            "section": route_section(float(body[:, 0].mean()), self.route),
            "seen_slope_support": self.seen_slope_support,
            "seen_stair_support": self.seen_stair_support,
            "physical_failure": self.physical_failure,
            "success": self.success and not self.physical_failure,
        }


def update_dwell_torch(
    *,
    route: RouteSpec,
    acceptance: AcceptanceSpec,
    prior_dwell_s,
    wheel_xy,
    body_hull_xy,
    wheel_support,
    seen_slope_support,
    seen_stair_support,
    translation_speed_mps,
    yaw_rate_rps,
    body_tilt_rad,
    physical_failure,
    dt_s: float,
):
    """Vectorized GPU step; caller checkpoints the dwell and seen-section buffers.

    Inputs are torch tensors for N environments in tile-local XY.  Physical
    failure is a per-environment boolean. The caller must never use V23's 4 m
    radius or stair_route_exit as a substitute for this complete-route gate.
    """
    import torch

    if (wheel_xy.ndim != 3 or wheel_xy.shape[1:] != (4, 2)
            or body_hull_xy.ndim != 3 or body_hull_xy.shape[0] != len(wheel_xy)
            or body_hull_xy.shape[1] < 4 or body_hull_xy.shape[2] != 2
            or wheel_support.shape != wheel_xy.shape[:2]
            or prior_dwell_s.shape != (len(wheel_xy),)):
        raise ValueError("Expected aligned [N,4,2] wheels and [N,B,2] body hull")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Control-step duration must be positive and finite")
    supported_x = wheel_xy[..., 0]
    slope = ((supported_x >= route.approach_end_x_m)
             & (supported_x < route.slope_end_x_m) & wheel_support).any(dim=1)
    stair = ((supported_x >= route.transition_end_x_m)
             & (supported_x < route.final_start_x_m) & wheel_support).any(dim=1)
    seen_slope_support = seen_slope_support | slope
    seen_stair_support = seen_stair_support | stair
    points = torch.cat((wheel_xy, body_hull_xy), dim=1)
    low = torch.tensor((route.final_start_x_m + acceptance.edge_margin_m,
                        route.tile_border_m + acceptance.border_margin_m),
                       dtype=points.dtype, device=points.device)
    high = torch.tensor((route.size_x_m - route.tile_border_m - acceptance.border_margin_m,
                         route.size_y_m - route.tile_border_m - acceptance.border_margin_m),
                        dtype=points.dtype, device=points.device)
    on_final = ((points >= low) & (points <= high)).all(dim=(1, 2))
    stable = (on_final & wheel_support.all(dim=1) & seen_slope_support & seen_stair_support
              & (translation_speed_mps.abs() <= acceptance.max_translation_mps)
              & (yaw_rate_rps.abs() <= acceptance.max_yaw_rps)
              & (body_tilt_rad.abs() <= acceptance.max_tilt_rad) & ~physical_failure)
    dwell_s = torch.where(stable, prior_dwell_s + dt_s, torch.zeros_like(prior_dwell_s))
    return {
        "dwell_s": dwell_s,
        # Float32 accumulation of 100 x 0.02 s can undershoot 2.0 by a few
        # microseconds. Tolerance is only 0.1% of one control step, so this
        # accepts the intended 100th sample without admitting the 99th.
        "success": dwell_s + dt_s * 1e-3 >= acceptance.dwell_required_s,
        "stable_now": stable,
        "all_wheels_and_body_on_final_platform": on_final,
        "seen_slope_support": seen_slope_support,
        "seen_stair_support": seen_stair_support,
    }
