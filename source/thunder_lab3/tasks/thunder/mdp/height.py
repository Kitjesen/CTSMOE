"""Stateful height-reference candidate; independent of a physics backend."""
from dataclasses import dataclass, replace
import math

import torch


@dataclass
class HeightOptions:
    # Lab 3 configclass recursively assigns fields while resolving config values.
    target_height: float
    tolerance: float
    max_reference_speed: float
    max_reference_lag: float
    min_valid_fraction: float = 1.0

    def __post_init__(self):
        values = (self.target_height, self.tolerance, self.max_reference_speed,
                  self.max_reference_lag, self.min_valid_fraction)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("height settings must be finite")
        if self.target_height <= 0 or not 0 <= self.tolerance < self.target_height:
            raise ValueError("require target_height > tolerance >= 0")
        if self.max_reference_speed <= 0 or self.max_reference_lag <= 0:
            raise ValueError("reference speed and lag must be positive")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")


class HeightReference:
    """Track a cropped ground estimate; return an unweighted squared height cost.

    During a scan discontinuity the reference is rate-limited and the cost is
    suppressed while it lags too far. This does not estimate a support plane.
    """
    def __init__(self, count, device, dtype, dt, options):
        if count <= 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("count and dt must be positive")
        self.options = replace(options)  # Revalidate composed values and isolate runtime state.
        self.max_step = self.options.max_reference_speed * dt
        self.ground_z = torch.zeros(count, device=device, dtype=dtype)
        self.initialized = torch.zeros(count, device=device, dtype=torch.bool)
        self.active = torch.zeros_like(self.initialized)
        self.lag = torch.zeros_like(self.ground_z)
        self.relative_height = torch.zeros_like(self.ground_z)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.ground_z[ids] = 0
        self.initialized[ids] = False
        self.active[ids] = False
        self.lag[ids] = 0
        self.relative_height[ids] = 0

    def step(self, base_z, hits_z):
        if base_z.shape != self.ground_z.shape or hits_z.ndim != 2:
            raise ValueError("expected base_z [N], hits_z [N, rays]")
        if hits_z.shape[0] != len(base_z) or hits_z.shape[1] == 0:
            raise ValueError("height scan requires at least one ray per environment")
        if base_z.device != self.ground_z.device or hits_z.device != base_z.device:
            raise ValueError("state, base, and rays must use the same device")
        finite = torch.isfinite(hits_z)
        count = finite.sum(dim=1)
        enough = count >= math.ceil(hits_z.shape[1] * self.options.min_valid_fraction)
        valid = enough & torch.isfinite(base_z)
        raw = torch.where(finite, hits_z, 0).sum(dim=1) / count.clamp_min(1)
        first = valid & ~self.initialized
        previous = self.ground_z
        limited = previous + torch.clamp(raw - previous, -self.max_step, self.max_step)
        self.ground_z = torch.where(first, raw, torch.where(valid, limited, previous))
        self.initialized |= valid
        self.lag = torch.where(valid, torch.abs(raw - self.ground_z), 0)
        self.active = valid & (self.lag <= self.options.max_reference_lag)
        # Missing data never produces a fabricated height penalty.
        self.relative_height = torch.where(valid, base_z - self.ground_z, 0)
        error = (torch.abs(self.relative_height - self.options.target_height)
                 - self.options.tolerance).clamp_min(0)
        return torch.where(self.active, error.square(), 0)
