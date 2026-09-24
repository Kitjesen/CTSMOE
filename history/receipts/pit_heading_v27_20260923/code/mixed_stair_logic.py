"""CPU/GPU tensor rules for a bounded mixed-command stair curriculum.

This module is deliberately independent of Isaac Lab. The host must provide
measured terminal-platform geometry and either four terrain-contact points or
a separately identified conservative collision-hull support witness. It must
not infer completion from the old four-metre radial distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


PROFILE = "mixed_stair_short_pulses_v1"
STRAIGHT, YAW, LATERAL = range(3)
GROUP_NAMES = ("straight", "forward_yaw", "forward_lateral")
GROUP_WEIGHTS = (0.8, 0.1, 0.1)
SPEED_EDGES = (0.2, 0.4, 0.6, 0.8, 1.0)
PULSE_BOUNDARIES_S = (2.0, 2.8, 3.0, 3.6, 4.0)
YAW_RATE_RAD_S = 0.12
LATERAL_SPEED_M_S = 0.08
EPISODE_S = 25.0
CONTROL_DT_S = 0.02
STATS_FIELDS = (
    "episodes", "eligible", "completed", "promoted", "downgraded",
    "physical_failure", "lateral_error_sum_m", "vx_mae_sum_mps",
    "vy_mae_sum_mps", "wz_mae_sum_rad_s",
)


@dataclass(frozen=True)
class StairGeometry:
    """Measured final plateau in coordinates measured outward from tile centre."""

    platform_start_m: float
    platform_end_m: float
    lane_half_width_m: float
    platform_z_mesh_m: float
    row: int
    column: int
    direction: str
    axis: str

    @classmethod
    def from_coverage_receipt(cls, receipt: dict[str, Any], axis: str) -> StairGeometry:
        if not receipt.get("selected") or receipt.get("kind") not in ("stairs_up", "stairs_down"):
            raise ValueError("Mixed stairs require a selected, measured terminal-platform cell")
        if axis not in ("positive_x", "negative_x", "positive_y", "negative_y"):
            raise ValueError("Unknown outward stair axis")
        runs = receipt["geometry_after"][axis]["surface_runs"]
        if len(runs) < 3:
            raise ValueError("Measured route does not contain a flight and final plateau")
        last = runs[-1]
        start, end, z = float(last["from_m"]), float(last["to_m"]), float(last["z_m"])
        if not all(map(math.isfinite, (start, end, z))) or not (2.5 < start < 3.5 and 4.4 < end <= 4.6):
            raise ValueError("Terminal plateau geometry changed; remeasure the route")
        if end - start < 1.0 or abs(z - float(runs[0]["z_m"])) < 0.02:
            raise ValueError("Selected cell lacks a distinct usable terminal plateau")
        # The measured centre platform is 3 m wide.  One metre leaves space
        # for the robot footprint and a margin on both sides.
        return cls(start, end, 1.0, z, int(receipt["row"]), int(receipt["column"]),
                   receipt["kind"], axis)


def sample_cohorts(count: int, device: torch.device | str, generator: torch.Generator | None = None):
    """Return group, sign, speed bin, and positive forward speed for new episodes."""
    if count < 0:
        raise ValueError("Negative episode count")
    draw = torch.rand(count, device=device, generator=generator)
    group = torch.where(draw < GROUP_WEIGHTS[0], STRAIGHT,
                        torch.where(draw < sum(GROUP_WEIGHTS[:2]), YAW, LATERAL)).long()
    side = torch.where(torch.rand(count, device=device, generator=generator) < 0.5, -1, 1)
    sign = torch.where(group == STRAIGHT, 0, side).long()
    speed_bin = torch.randint(0, 4, (count,), device=device, generator=generator)
    edges = torch.tensor(SPEED_EDGES, device=device)
    speed = edges[speed_bin] + torch.rand(count, device=device, generator=generator) * (
        edges[speed_bin + 1] - edges[speed_bin])
    # Mixed commands are trained first where the 25 s episode can reach the
    # measured final platform without a high-speed cornering demand.
    mixed = group != STRAIGHT
    speed = torch.where(mixed, 0.4 + 0.4 * torch.rand(count, device=device, generator=generator), speed)
    speed_bin = torch.bucketize(speed.contiguous(), edges[1:-1].contiguous(), right=True).long()
    return group, sign, speed_bin, speed


def scheduled_command(group: torch.Tensor, sign: torch.Tensor, vx: torch.Tensor,
                      elapsed_s: torch.Tensor) -> torch.Tensor:
    """Body-frame [vx, vy, yaw-rate], with a balanced short correction pulse."""
    if not (group.shape == sign.shape == vx.shape == elapsed_s.shape):
        raise ValueError("Cohort, sign, speed and time must have the same shape")
    yaw = torch.where((group == YAW) & (elapsed_s >= 2.0) & (elapsed_s < 2.8),
                      sign * YAW_RATE_RAD_S, torch.zeros_like(vx))
    yaw = torch.where((group == YAW) & (elapsed_s >= 2.8) & (elapsed_s < 3.6),
                      -sign * YAW_RATE_RAD_S, yaw)
    vy = torch.where((group == LATERAL) & (elapsed_s >= 2.0) & (elapsed_s < 3.0),
                     sign * LATERAL_SPEED_M_S, torch.zeros_like(vx))
    vy = torch.where((group == LATERAL) & (elapsed_s >= 3.0) & (elapsed_s < 4.0),
                     -sign * LATERAL_SPEED_M_S, vy)
    return torch.stack((vx, vy, yaw), dim=-1)


def integrate_body_twist(pose: torch.Tensor, command: torch.Tensor, dt_s: torch.Tensor) -> torch.Tensor:
    """Exact constant-twist SE(2) step; pose/command have shape (N,3)."""
    if pose.shape != command.shape or pose.ndim != 2 or pose.shape[1] != 3 or dt_s.shape != pose.shape[:1]:
        raise ValueError("Expected pose/command (N,3) and dt (N,)")
    vx, vy, wz = command.unbind(-1)
    angle = wz * dt_s
    safe_wz = torch.where(wz.abs() > 1e-8, wz, torch.ones_like(wz))
    dx = torch.where(wz.abs() > 1e-8,
                     (vx * torch.sin(angle) + vy * (torch.cos(angle) - 1)) / safe_wz,
                     vx * dt_s)
    dy = torch.where(wz.abs() > 1e-8,
                     (vx * (1 - torch.cos(angle)) + vy * torch.sin(angle)) / safe_wz,
                     vy * dt_s)
    heading = pose[:, 2]
    return torch.stack((pose[:, 0] + dx * torch.cos(heading) - dy * torch.sin(heading),
                        pose[:, 1] + dx * torch.sin(heading) + dy * torch.cos(heading),
                        heading + angle), dim=-1)


def integrate_schedule(pose: torch.Tensor, group: torch.Tensor, sign: torch.Tensor,
                       speed: torch.Tensor, elapsed_s: torch.Tensor, dt_s: float) -> torch.Tensor:
    """Integrate the *issued* command, splitting steps that cross pulse edges."""
    if dt_s < 0 or not math.isfinite(dt_s):
        raise ValueError("Invalid control interval")
    end = elapsed_s + dt_s
    cuts = [elapsed_s]
    for boundary in PULSE_BOUNDARIES_S:
        cuts.append(torch.clamp(torch.full_like(elapsed_s, boundary), min=elapsed_s, max=end))
    cuts.append(end)
    result = pose
    for left, right in zip(cuts[:-1], cuts[1:]):
        result = integrate_body_twist(result, scheduled_command(group, sign, speed, left), right - left)
    return result


def planned_route_is_in_lane(geometry: StairGeometry, group: int, sign: int, speed: float,
                             dt_s: float = CONTROL_DT_S) -> bool:
    """Check command intent until a plausible full-body platform crossing."""
    if not (0.2 <= speed <= 1.0 and sign in (-1, 0, 1)):
        return False
    if group != STRAIGHT and not 0.4 <= speed < 0.8:
        return False
    if group == STRAIGHT and sign != 0 or group != STRAIGHT and sign == 0:
        return False
    pose = torch.zeros((1, 3), dtype=torch.float64)
    group_t = torch.tensor([group])
    sign_t = torch.tensor([sign])
    speed_t = torch.tensor([speed], dtype=torch.float64)
    target = geometry.platform_start_m + 0.50
    for step in range(math.ceil(EPISODE_S / dt_s)):
        pose = integrate_schedule(pose, group_t, sign_t, speed_t,
                                  torch.tensor([step * dt_s], dtype=torch.float64), dt_s)
        x, y, yaw = pose[0].tolist()
        if abs(y) > geometry.lane_half_width_m or abs(yaw) > 0.15 or x > geometry.platform_end_m - 0.20:
            return False
        if x >= target:
            return True
    return False


def signed_path_error(actual_local_pose: torch.Tensor, desired_local_pose: torch.Tensor) -> torch.Tensor:
    """Command-relative displacement/heading error, not absolute sideways motion."""
    if actual_local_pose.shape != desired_local_pose.shape or actual_local_pose.shape[-1] != 3:
        raise ValueError("Expected corresponding local SE(2) poses")
    error = actual_local_pose - desired_local_pose
    error[..., 2] = torch.atan2(torch.sin(error[..., 2]), torch.cos(error[..., 2]))
    return error


def actual_pose_in_stair_frame(root_world_xy: torch.Tensor, root_quat_wxyz: torch.Tensor,
                               tile_origin_world_xy: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Project actual root pose onto its reset cardinal stair axis."""
    if (root_world_xy.ndim != 2 or root_world_xy.shape[1] != 2
            or root_quat_wxyz.shape != (len(axis), 4)
            or tile_origin_world_xy.shape != root_world_xy.shape or axis.shape != root_world_xy.shape[:1]):
        raise ValueError("Expected world XY, wxyz root quaternion, tile origin, and cardinal axis")
    if torch.any((axis < 0) | (axis > 3)):
        raise ValueError("Invalid cardinal stair axis")
    heading = axis * (math.pi / 2)
    delta = root_world_xy - tile_origin_world_xy
    x = delta[:, 0] * torch.cos(heading) + delta[:, 1] * torch.sin(heading)
    y = -delta[:, 0] * torch.sin(heading) + delta[:, 1] * torch.cos(heading)
    w, qx, qy, qz = root_quat_wxyz.unbind(-1)
    yaw = torch.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy.square() + qz.square())) - heading
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.stack((x, y, yaw), -1)


@dataclass
class PlatformWitness:
    """Consecutive observations that all four wheels reached the final platform."""

    geometry: StairGeometry
    required_steps: int = 10
    contact_threshold_n: float = 5.0
    margin_m: float = 0.10
    streak: int = 0

    def observe(self, support_points_local_xy: torch.Tensor, terrain_contact_force_n: torch.Tensor,
                physical_failure: bool = False) -> bool:
        if support_points_local_xy.shape != (4, 2) or terrain_contact_force_n.shape != (4,):
            raise ValueError("Require four terrain-contact points and four terrain-filtered forces")
        if not torch.isfinite(support_points_local_xy).all() or not torch.isfinite(terrain_contact_force_n).all():
            raise ValueError("Nonfinite support observation")
        forward, lateral = support_points_local_xy[:, 0], support_points_local_xy[:, 1]
        on_platform = bool(((forward >= self.geometry.platform_start_m + self.margin_m)
                            & (forward <= self.geometry.platform_end_m - self.margin_m)
                            & (lateral.abs() <= self.geometry.lane_half_width_m)
                            & (terrain_contact_force_n >= self.contact_threshold_n)).all())
        self.streak = self.streak + 1 if on_platform and not physical_failure else 0
        return self.streak >= self.required_steps


def conservative_center_hull_support(
    wheel_centers_local_xy: torch.Tensor, body_hull_local_xy: torch.Tensor,
    wheel_center_world_z: torch.Tensor, platform_world_z: torch.Tensor,
    terrain_contact_force_n: torch.Tensor, start_m: torch.Tensor, end_m: torch.Tensor,
    lane_half_width_m: torch.Tensor, wheel_radius_m: float, *, margin_m: float = 0.10,
    height_tolerance_m: float = 0.08,
) -> torch.Tensor:
    """Conservative *inferred* platform support when contact XY is unavailable.

    All physical wheel collider centres are inset by one wheel radius plus a
    margin.  Every ground-projected body collision-hull corner is past the
    final riser.  Four terrain-filtered wheel forces must be present.  This
    establishes a bounded geometric witness, not measured contact locations.
    """
    count = len(start_m)
    if (wheel_centers_local_xy.shape != (count, 4, 2)
            or body_hull_local_xy.ndim != 3 or body_hull_local_xy.shape[0] != count
            or body_hull_local_xy.shape[1] < 4 or body_hull_local_xy.shape[2] != 2
            or wheel_center_world_z.shape != (count, 4) or platform_world_z.shape != (count,)
            or terrain_contact_force_n.shape != (count, 4)
            or end_m.shape != (count,) or lane_half_width_m.shape != (count,)):
        raise ValueError("Require four wheel collider centres, collision hull and four terrain forces")
    if (not all(torch.isfinite(value).all() for value in
                (wheel_centers_local_xy, body_hull_local_xy, wheel_center_world_z,
                 platform_world_z, terrain_contact_force_n,
                 start_m, end_m, lane_half_width_m))
            or not (0 < wheel_radius_m < 0.30 and 0 < margin_m < 0.30
                    and 0 < height_tolerance_m < wheel_radius_m)):
        raise ValueError("Invalid collision geometry or nonfinite support data")
    inset = wheel_radius_m + margin_m
    if torch.any((end_m - start_m <= 2 * inset) | (lane_half_width_m <= inset)):
        raise ValueError("Platform too narrow for the claimed collision envelope")
    wheel_x, wheel_y = wheel_centers_local_xy.unbind(-1)
    hull_x, hull_y = body_hull_local_xy.unbind(-1)
    wheel_inside = ((wheel_x >= start_m[:, None] + inset)
                    & (wheel_x <= end_m[:, None] - inset)
                    & (wheel_y.abs() <= lane_half_width_m[:, None] - inset)
                    & ((wheel_center_world_z - wheel_radius_m - platform_world_z[:, None]).abs()
                       <= height_tolerance_m)
                    & (terrain_contact_force_n >= 5.)).all(-1)
    hull_inside = ((hull_x >= start_m[:, None] + margin_m)
                   & (hull_x <= end_m[:, None] - margin_m)
                   & (hull_y.abs() <= lane_half_width_m[:, None])).all(-1)
    return wheel_inside & hull_inside


def curriculum_decision(*, completed_platform: bool, eligible: bool, physical_failure: bool,
                        lateral_error_m: float, yaw_error_rad: float,
                        vx_mae_mps: float, vy_mae_mps: float, wz_mae_rad_s: float) -> tuple[bool, bool]:
    """Return promote/downgrade for a complete-flight command category."""
    if physical_failure:
        return False, True
    if not eligible:
        return False, False
    tracking_ok = (abs(lateral_error_m) <= 0.35 and abs(yaw_error_rad) <= 0.25
                   and vx_mae_mps <= 0.25 and vy_mae_mps <= 0.15 and wz_mae_rad_s <= 0.15)
    promoted = completed_platform and tracking_ok
    return promoted, not promoted


class MixedStairCourse:
    """Serializable per-environment command state and category/sign statistics."""

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        if num_envs <= 0:
            raise ValueError("Expected positive environment count")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.group = torch.full((num_envs,), STRAIGHT, dtype=torch.long, device=device)
        self.sign = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.speed_bin = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.speed = torch.zeros(num_envs, device=device)
        self.elapsed_s = torch.zeros(num_envs, device=device)
        self.desired_pose = torch.zeros((num_envs, 3), device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.support_streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.platform_complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.tracking_error_sum = torch.zeros((num_envs, 3), device=device)
        self.tracking_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Each direction of a brief yaw/lateral pulse is checked separately.
        # Full-episode MAE would dilute an ignored pulse over the 25 s rollout.
        self.pulse_count = torch.zeros((num_envs, 2), device=device, dtype=torch.long)
        self.pulse_response_sum = torch.zeros((num_envs, 2), device=device)
        self.pulse_abs_error_sum = torch.zeros((num_envs, 2), device=device)
        # learner, up/down, terrain level, group, sign+1, speed bin, fields
        self.episode_totals = torch.zeros((2, 2, 10, 3, 3, 4, len(STATS_FIELDS)),
                                          dtype=torch.float64, device=device)

    def start(self, ids: torch.Tensor, group: torch.Tensor, sign: torch.Tensor,
              speed_bin: torch.Tensor, speed: torch.Tensor,
              initial_local_pose: torch.Tensor | None = None) -> None:
        if not (ids.shape == group.shape == sign.shape == speed_bin.shape == speed.shape):
            raise ValueError("Episode cohort arrays differ")
        if (torch.any((group < 0) | (group > 2)) or torch.any((sign < -1) | (sign > 1))
                or torch.any((group == STRAIGHT) != (sign == 0)) or torch.any((speed_bin < 0) | (speed_bin > 3))
                or torch.any((speed < SPEED_EDGES[0]) | (speed > SPEED_EDGES[-1]))):
            raise ValueError("Invalid stair cohort")
        self.group[ids], self.sign[ids], self.speed_bin[ids], self.speed[ids] = group, sign, speed_bin, speed
        self.elapsed_s[ids] = 0
        if initial_local_pose is not None and initial_local_pose.shape != (len(ids), 3):
            raise ValueError("Expected one initial stair-local pose per reset environment")
        self.desired_pose[ids] = 0 if initial_local_pose is None else initial_local_pose
        self.active[ids] = True
        self.support_streak[ids] = 0
        self.platform_complete[ids] = False
        self.tracking_error_sum[ids] = 0
        self.tracking_count[ids] = 0
        self.pulse_count[ids] = 0
        self.pulse_response_sum[ids] = 0
        self.pulse_abs_error_sum[ids] = 0

    def advance(self, ids: torch.Tensor, dt_s: float) -> torch.Tensor:
        old_time = self.elapsed_s[ids]
        command = scheduled_command(self.group[ids], self.sign[ids], self.speed[ids], old_time)
        self.desired_pose[ids] = integrate_schedule(self.desired_pose[ids], self.group[ids],
                                                    self.sign[ids], self.speed[ids], old_time, dt_s)
        self.elapsed_s[ids] += dt_s
        return command

    def observe_tracking(self, ids: torch.Tensor, actual_body_velocity: torch.Tensor,
                         issued_command: torch.Tensor) -> None:
        if actual_body_velocity.shape != issued_command.shape or actual_body_velocity.shape != (len(ids), 3):
            raise ValueError("Expected corresponding (N,3) body velocities and commands")
        self.tracking_error_sum[ids] += (actual_body_velocity - issued_command).abs()
        self.tracking_count[ids] += 1
        group = self.group[ids]
        elapsed = self.elapsed_s[ids]
        axis = torch.where(group == YAW, 2, 1)
        actual_axis = actual_body_velocity.gather(1, axis[:, None]).squeeze(1)
        command_axis = issued_command.gather(1, axis[:, None]).squeeze(1)
        split = torch.where(group == YAW, 2.8, 3.0)
        for phase in (0, 1):
            selected = (group != STRAIGHT) & (command_axis.abs() > 1e-6)
            selected &= (elapsed < split) if phase == 0 else (elapsed >= split)
            picked = ids[selected]
            if len(picked):
                self.pulse_count[picked, phase] += 1
                self.pulse_response_sum[picked, phase] += (
                    actual_axis[selected] * command_axis[selected].sign())
                self.pulse_abs_error_sum[picked, phase] += (
                    actual_axis[selected] - command_axis[selected]).abs()

    def pulse_followed(self, ids: torch.Tensor) -> torch.Tensor:
        """Require a real same-direction response during *both* command pulses."""
        group = self.group[ids]
        count = self.pulse_count[ids]
        response = self.pulse_response_sum[ids] / count.clamp_min(1)
        error = self.pulse_abs_error_sum[ids] / count.clamp_min(1)
        amplitude = torch.where(group == YAW, YAW_RATE_RAD_S, LATERAL_SPEED_M_S)
        minimum_count = torch.where(group == YAW, 20, 25)
        followed = ((count >= minimum_count[:, None])
                    & (response >= .40 * amplitude[:, None])
                    & (error <= .75 * amplitude[:, None])).all(-1)
        return torch.where(group == STRAIGHT, torch.ones_like(followed), followed)

    def record(self, ids: torch.Tensor, learner: torch.Tensor, direction: torch.Tensor,
               level: torch.Tensor, eligible: torch.Tensor, completed: torch.Tensor,
               promoted: torch.Tensor, downgraded: torch.Tensor, physical_failure: torch.Tensor,
               lateral_error_m: torch.Tensor, tracking_mae: torch.Tensor) -> None:
        if (torch.any((learner < 0) | (learner > 1)) or torch.any((direction < 0) | (direction > 1))
                or torch.any((level < 0) | (level > 9))):
            raise ValueError("Episode statistic index outside curriculum")
        if tracking_mae.shape != (len(ids), 3):
            raise ValueError("Expected per-axis tracking MAE")
        values = torch.cat((torch.stack((torch.ones_like(eligible), eligible, completed, promoted,
                                         downgraded, physical_failure, lateral_error_m.abs()), -1).double(),
                            tracking_mae.double()), -1)
        key = (((((learner * 2 + direction) * 10 + level) * 3 + self.group[ids]) * 3
                 + self.sign[ids] + 1) * 4 + self.speed_bin[ids])
        self.episode_totals.view(-1, len(STATS_FIELDS)).index_add_(0, key, values)

    def observe_platform(self, ids: torch.Tensor, support_points_local_xy: torch.Tensor,
                         terrain_contact_force_n: torch.Tensor, start_m: torch.Tensor,
                         end_m: torch.Tensor, lane_half_width_m: torch.Tensor,
                         physical_failure: torch.Tensor, *, required_steps: int = 10) -> torch.Tensor:
        """Check four measured contact points per env; no inferred wheel centres."""
        count = len(ids)
        if (support_points_local_xy.shape != (count, 4, 2)
                or terrain_contact_force_n.shape != (count, 4)
                or any(value.shape != (count,) for value in
                       (start_m, end_m, lane_half_width_m, physical_failure))):
            raise ValueError("Missing four terrain-contact points or measured platform bounds")
        if (not torch.isfinite(support_points_local_xy).all()
                or not torch.isfinite(terrain_contact_force_n).all()):
            raise ValueError("Nonfinite terrain contact observation")
        forward, lateral = support_points_local_xy.unbind(-1)
        supported = ((forward >= start_m[:, None] + 0.10)
                     & (forward <= end_m[:, None] - 0.10)
                     & (lateral.abs() <= lane_half_width_m[:, None])
                     & (terrain_contact_force_n >= 5.0)).all(-1) & ~physical_failure
        return self._record_support(ids, supported, required_steps)

    def observe_platform_conservative(
        self, ids: torch.Tensor, wheel_centers_local_xy: torch.Tensor,
        body_hull_local_xy: torch.Tensor, wheel_center_world_z: torch.Tensor,
        platform_world_z: torch.Tensor, terrain_contact_force_n: torch.Tensor,
        start_m: torch.Tensor, end_m: torch.Tensor, lane_half_width_m: torch.Tensor,
        wheel_radius_m: float, physical_failure: torch.Tensor, *, required_steps: int = 10,
    ) -> torch.Tensor:
        supported = conservative_center_hull_support(
            wheel_centers_local_xy, body_hull_local_xy, wheel_center_world_z,
            platform_world_z, terrain_contact_force_n,
            start_m, end_m, lane_half_width_m, wheel_radius_m) & ~physical_failure
        return self._record_support(ids, supported, required_steps)

    def _record_support(self, ids: torch.Tensor, supported: torch.Tensor,
                        required_steps: int) -> torch.Tensor:
        if supported.shape != ids.shape or required_steps < 1:
            raise ValueError("Invalid support streak input")
        self.support_streak[ids] = torch.where(supported, self.support_streak[ids] + 1, 0)
        self.platform_complete[ids] |= self.support_streak[ids] >= required_steps
        return self.platform_complete[ids]

    def state_dict(self) -> dict[str, Any]:
        return {"profile": PROFILE, "num_envs": self.num_envs,
                **{name: getattr(self, name).detach().cpu().clone()
                   for name in ("group", "sign", "speed_bin", "speed", "elapsed_s",
                                "desired_pose", "active", "support_streak", "platform_complete",
                                "tracking_error_sum", "tracking_count", "pulse_count",
                                "pulse_response_sum", "pulse_abs_error_sum",
                                "episode_totals")}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("profile") != PROFILE or state.get("num_envs") != self.num_envs:
            raise ValueError("Mixed stair profile or environment count changed")
        for name in ("group", "sign", "speed_bin", "speed", "elapsed_s", "desired_pose",
                     "active", "support_streak", "platform_complete", "tracking_error_sum",
                     "tracking_count", "pulse_count", "pulse_response_sum",
                     "pulse_abs_error_sum", "episode_totals"):
            saved = state[name]
            target = getattr(self, name)
            if saved.shape != target.shape or saved.dtype != target.dtype or not torch.isfinite(saved).all():
                raise ValueError(f"Invalid saved stair field: {name}")
            target.copy_(saved.to(self.device))
        if (torch.any((self.group < 0) | (self.group > 2)) or torch.any((self.sign < -1) | (self.sign > 1))
                or torch.any((self.group == STRAIGHT) != (self.sign == 0))
                or torch.any((self.speed_bin < 0) | (self.speed_bin > 3))
                or torch.any(self.elapsed_s < 0) or torch.any(self.support_streak < 0)
                or torch.any(self.pulse_count < 0) or torch.any(self.pulse_abs_error_sum < 0)
                or torch.any(self.episode_totals < 0)):
            raise ValueError("Saved mixed stair state violates the curriculum contract")
