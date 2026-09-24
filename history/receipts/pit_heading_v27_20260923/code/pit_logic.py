"""Tensor success rules and serializable, student-gated pit height curriculum."""
from dataclasses import asdict, dataclass, field
import math

import numpy as np
import torch


def top_support(wheel_pos, wheel_force, edge_x, top_z, radius=0.093):
    """Reject wall contact and airborne crossings; wheel centers are joint origins."""
    past_edge = wheel_pos[..., 0] - radius >= edge_x[:, None] + 0.05
    near_top = (wheel_pos[..., 2] - radius - top_z[:, None]).abs() < 0.035
    return past_edge & near_top & (wheel_force[..., 2] > 5.0)


def stable_success(hold, supported, body_clear, upright, quiet, violation, started_in_pit, dt):
    """Count control steps exactly, avoiding floating point drift at two seconds."""
    valid = supported.all(-1) & body_clear & upright & quiet & ~violation & started_in_pit
    hold = torch.where(valid, hold + 1, torch.zeros_like(hold))
    return hold, hold >= math.ceil(2.0 / dt)


@dataclass
class PitCourse:
    """Training-episode gate; independent held-out evaluation remains a separate report.

    All speed bins need two disjoint batches of 64 student frontier episodes with
    >=90% complete successes and <=5% physical failures before a 5 cm increase.
    """
    frontier: int = 2
    max_level: int = 21
    min_samples: int = 64
    streak: int = 0
    window: list = field(default_factory=lambda: np.zeros((4, 3), dtype=int).tolist())
    decisions: list = field(default_factory=list)

    def observe(self, bins, success, physical):
        a = np.asarray(self.window, dtype=np.int64)
        for b, ok, fail in zip(bins, success, physical, strict=True):
            if not 0 <= b < 4 or (ok and fail):
                raise ValueError("Invalid curriculum episode")
            a[b] += [1, int(ok), int(fail)]
        self.window = a.tolist()
        if (a[:, 0] < self.min_samples).any():
            return False
        passed = bool(((a[:, 1]/a[:, 0] >= 0.9) & (a[:, 2]/a[:, 0] <= 0.05)).all())
        self.streak = self.streak + 1 if passed else 0
        advance = self.streak >= 2 and self.frontier < self.max_level
        self.decisions.append(dict(level=self.frontier, counts=a.tolist(), passed=passed, advance=advance))
        self.window = np.zeros((4, 3), dtype=int).tolist()
        if advance:
            self.frontier += 1
            self.streak = 0
        return advance

    def state_dict(self):
        return asdict(self)

    @classmethod
    def restore(cls, state):
        result = cls(**state)
        if not 0 <= result.frontier <= result.max_level <= 21:
            raise ValueError("Invalid saved pit frontier")
        if np.asarray(result.window).shape != (4, 3):
            raise ValueError("Invalid saved pit counts")
        return result
