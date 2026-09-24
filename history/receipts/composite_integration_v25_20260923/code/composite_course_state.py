"""Tensor state for a composite-route acceptance course.

This class is independent of Isaac sensors. The caller must supply current
wheel contacts and actual full-body collision points in route-local XY.
"""

from __future__ import annotations

import torch

from composite_route import COMPACT_9X9
from route_acceptance import AcceptanceSpec, update_dwell_torch

PROFILE = "composite_course_state_v2"
STAT_NAMES = ("episodes", "slope_support", "stair_support", "full_body_platform",
              "completed", "failed_or_timeout")


class CompositeCourseState:
    def __init__(self, num_envs: int, device="cpu", acceptance=None):
        self.num_envs = num_envs
        self.device = device
        self.acceptance = acceptance or AcceptanceSpec()
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.seen_slope = self.active.clone()
        self.seen_stair = self.active.clone()
        self.full_body_platform = self.active.clone()
        self.success = self.active.clone()
        self.dwell_s = torch.zeros(num_envs, device=device)
        self.totals = torch.zeros((2, len(STAT_NAMES)), dtype=torch.float64, device=device)

    def set_active(self, active):
        if active.shape != self.active.shape or active.dtype != torch.bool:
            raise ValueError("Composite terrain membership must be one boolean per environment")
        departing = self.active & ~active
        self.reset(departing.nonzero().flatten())
        self.active.copy_(active)

    def reset(self, ids):
        for value in (self.seen_slope, self.seen_stair, self.full_body_platform, self.success):
            value[ids] = False
        self.dwell_s[ids] = 0

    def update(self, *, wheel_xy, body_hull_xy, wheel_support,
               translation_speed_mps, yaw_rate_rps, body_tilt_rad,
               physical_failure, dt_s):
        result = update_dwell_torch(
            route=COMPACT_9X9, acceptance=self.acceptance,
            prior_dwell_s=self.dwell_s, wheel_xy=wheel_xy,
            body_hull_xy=body_hull_xy, wheel_support=wheel_support,
            seen_slope_support=self.seen_slope,
            seen_stair_support=self.seen_stair,
            translation_speed_mps=translation_speed_mps,
            yaw_rate_rps=yaw_rate_rps, body_tilt_rad=body_tilt_rad,
            physical_failure=physical_failure, dt_s=dt_s,
        )
        self.seen_slope.copy_(result["seen_slope_support"] & self.active)
        self.seen_stair.copy_(result["seen_stair_support"] & self.active)
        self.full_body_platform.copy_(result["all_wheels_and_body_on_final_platform"] & self.active)
        self.dwell_s.copy_(torch.where(self.active, result["dwell_s"], 0))
        self.success |= result["success"] & self.active
        return self.success.clone()

    def update_subset(self, ids, *, wheel_xy, body_hull_xy, wheel_support,
                      translation_speed_mps, yaw_rate_rps, body_tilt_rad,
                      physical_failure, dt_s):
        """Evaluate only terminal candidates with the expensive collision hull."""
        if not len(ids):
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        result = update_dwell_torch(
            route=COMPACT_9X9, acceptance=self.acceptance,
            prior_dwell_s=self.dwell_s[ids], wheel_xy=wheel_xy,
            body_hull_xy=body_hull_xy, wheel_support=wheel_support,
            seen_slope_support=self.seen_slope[ids],
            seen_stair_support=self.seen_stair[ids],
            translation_speed_mps=translation_speed_mps,
            yaw_rate_rps=yaw_rate_rps, body_tilt_rad=body_tilt_rad,
            physical_failure=physical_failure, dt_s=dt_s,
        )
        self.seen_slope[ids] = result["seen_slope_support"]
        self.seen_stair[ids] = result["seen_stair_support"]
        self.full_body_platform[ids] = result["all_wheels_and_body_on_final_platform"]
        self.dwell_s[ids] = result["dwell_s"]
        self.success[ids] |= result["success"]
        return result["success"]

    def record_episodes(self, ids, learner, physical_failure):
        ids = ids[self.active[ids]]
        if not len(ids):
            return
        values = torch.stack((torch.ones_like(self.dwell_s[ids]),
                              self.seen_slope[ids], self.seen_stair[ids],
                              self.full_body_platform[ids], self.success[ids],
                              physical_failure[ids]), dim=1).to(torch.float64)
        self.totals.index_add_(0, learner[ids].long(), values)

    def state_dict(self):
        return {"profile": PROFILE, "num_envs": self.num_envs,
                **{name: getattr(self, name).clone() for name in (
                    "active", "seen_slope", "seen_stair", "full_body_platform",
                    "success", "dwell_s", "totals")}}

    def load_state_dict(self, saved, *, environment_state_restored: bool):
        if saved["profile"] != PROFILE or saved["num_envs"] != self.num_envs:
            raise ValueError("Incompatible composite curriculum checkpoint")
        self.totals.copy_(saved["totals"].to(self.device))
        if environment_state_restored:
            # Exact mid-episode recovery requires the simulation body/actuator,
            # contact, random-command and episode-clock state as well.
            for name in ("active", "seen_slope", "seen_stair", "full_body_platform",
                         "success", "dwell_s"):
                getattr(self, name).copy_(saved[name].to(self.device))
        else:
            # V23 checkpoint only saves model/optimizers and terrain assignment;
            # Isaac creates fresh physical episodes at resume. Preserve aggregate
            # statistics, then start honest new episode state after membership set.
            self.reset(torch.arange(self.num_envs, device=self.device))
