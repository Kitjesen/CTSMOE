"""Streaming runtime evidence for the V14 early-restart zero-update preflight."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


RESET_TARGETS_VERSION = "thunder_cts_v13_reset_targets_v1"


def timer_resampled(current: np.ndarray, previous: np.ndarray, dt: float) -> np.ndarray:
    return current > previous + dt * 0.5


def reset_count_payload(counts: torch.Tensor) -> dict[str, Any]:
    if counts.shape != (2, 10, 2) or counts.dtype != torch.long:
        raise RuntimeError(
            f"Expected long startup_reset_counts[2,10,2], got {counts.dtype} {tuple(counts.shape)}"
        )
    values = counts.detach().cpu().tolist()
    return {
        "axes": ["learner", "kind", "mode"],
        "counts": values,
        "all_cells_positive": all(
            v > 0 for learner in values for kind in learner for v in kind
        ),
    }


class Stats:
    def __init__(self, env: Any, alg: Any, output: Path):
        self.env, self.alg, self.output = env, alg, output
        self.command = env.command_manager.get_term("base_velocity")
        self.robot = env.scene["robot"]
        self.learner = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.learner[alg.student_env_idxs] = 1
        self.step = 0
        self.max_age = defaultdict(float)
        self.intervals = defaultdict(list)
        self.zero_translation_short_intervals = Counter()
        self.last_resample_age = np.full(env.num_envs, np.nan)
        self.last_translation_nonzero = np.zeros(env.num_envs, dtype=bool)
        self.previous_time_left = self.command.time_left.detach().cpu().numpy().copy()
        self.reward_nonfinite = 0
        self.action_nonfinite = 0
        self.command_observations = Counter()
        self._validate_configuration()
        receipt = getattr(env, "_thunder_reset_targets_installation", None)
        expected = {"version": RESET_TARGETS_VERSION, "robot": "robot"}
        if receipt != expected:
            raise RuntimeError(f"Missing reset target receipt: {receipt!r}")
        self.reset_target_receipt = dict(receipt)
        error = (
            (self.robot.data.joint_pos_target - self.robot.data.joint_pos).abs().max()
        )
        self.first_target_actual_max_abs_error = float(error.item())
        if self.first_target_actual_max_abs_error > 1e-6:
            raise RuntimeError(
                "First learn target does not equal actual reset joint pose"
            )

    def _validate_configuration(self) -> None:
        if getattr(self.env.cfg, "early_restart_profile", None) != "early_phase_v14":
            raise RuntimeError("early_phase_v14 is not active")
        for name in (
            "startup_repair_profile",
            "pit_posture_profile",
            "surface_posture_profile",
            "stance_strength_profile",
        ):
            if getattr(self.env.cfg, name, "off") != "off":
                raise RuntimeError(f"{name} must be explicitly off")
        if not np.isclose(
            self.env.max_episode_length * self.env.step_dt, 25.0, atol=1e-6
        ):
            raise RuntimeError("Expected 25 second episodes")
        if not np.isclose(float(self.command.cfg.resampling_time), 5.0, atol=1e-6):
            raise RuntimeError("Expected 5 second command resampling")
        reset_count_payload(self.command.startup_reset_counts)
        selected = self.command.startup_selected
        if any(
            int((selected & (self.learner == group)).sum()) <= 0 for group in (0, 1)
        ):
            raise RuntimeError("Both learner cohorts require selected command coverage")

    @torch.no_grad()
    def before(self, _obs: Any, actions: torch.Tensor) -> None:
        self.action_nonfinite += int((~torch.isfinite(actions)).sum())
        age = (self.env.episode_length_buf * self.env.step_dt).detach().cpu().numpy()
        learner = self.learner.detach().cpu().numpy()
        flat = (
            (self.command.terrain_idxs == self.command.terrain_type2idx["flat"])
            .detach()
            .cpu()
            .numpy()
        )
        for group in (0, 1):
            members = flat & (learner == group)
            if members.any():
                self.max_age[group] = max(
                    self.max_age[group], float(age[members].max())
                )
        current = self.command.time_left.detach().cpu().numpy().copy()
        events = timer_resampled(
            current, self.previous_time_left, float(self.env.step_dt)
        )
        active = self.command.startup_active.detach().cpu().numpy()
        translating = (
            (self.command.command[:, :2].abs().amax(dim=1) > 0.05)
            .detach()
            .cpu()
            .numpy()
        )
        for env_id in np.flatnonzero(events & flat & ~active & (age >= 4.9)):
            previous = self.last_resample_age[env_id]
            if np.isfinite(previous):
                interval = float(age[env_id] - previous)
                group = int(learner[env_id])
                if self.last_translation_nonzero[env_id] and translating[env_id]:
                    self.intervals[group].append(interval)
                elif interval < 4.9:
                    self.zero_translation_short_intervals[group] += 1
            self.last_resample_age[env_id] = age[env_id]
            self.last_translation_nonzero[env_id] = translating[env_id]
        self.previous_time_left = current
        selected = self.command.startup_selected
        kind = self.command.startup_kind
        for group in (0, 1):
            for command_kind in range(10):
                count = int(
                    (selected & (self.learner == group) & (kind == command_kind)).sum()
                )
                if count:
                    self.command_observations[(group, command_kind)] += count

    def after(
        self, _obs: Any, rewards: torch.Tensor, dones: torch.Tensor, _extras: Any
    ) -> None:
        self.reward_nonfinite += int((~torch.isfinite(rewards)).sum())
        done = dones.detach().cpu().numpy().astype(bool)
        self.last_resample_age[done] = np.nan
        self.last_translation_nonzero[done] = False
        self.step += 1

    def close(self) -> None:
        pass

    def finish(self) -> dict[str, Any]:
        counts = reset_count_payload(self.command.startup_reset_counts)
        intervals = {}
        for group, label in ((0, "teacher"), (1, "student")):
            values = self.intervals[group]
            intervals[label] = {
                "count": len(values),
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "short_intervals_involving_zero_translation": self.zero_translation_short_intervals[
                    group
                ],
            }
        summary = {
            "steps": self.step,
            "simulated_seconds": self.step * float(self.env.step_dt),
            "optimizer_steps": 0,
            "num_envs": self.env.num_envs,
            "configuration": {
                "episode_seconds": 25.0,
                "native_resampling_seconds": 5.0,
                "early_restart_profile": "early_phase_v14",
            },
            "maximum_episode_age_seconds": {
                "teacher": self.max_age[0],
                "student": self.max_age[1],
            },
            "native_resample_intervals_seconds": intervals,
            "reward_nonfinite_count": self.reward_nonfinite,
            "action_nonfinite_count": self.action_nonfinite,
            "reset_target_receipt": self.reset_target_receipt,
            "first_target_actual_max_abs_error": self.first_target_actual_max_abs_error,
            "startup_reset_counts": counts,
            "command_cohort_observations": {
                str(k): v for k, v in sorted(self.command_observations.items())
            },
        }
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "preflight_stats.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
