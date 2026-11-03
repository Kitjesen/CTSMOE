"""Streaming V13 sampler statistics for the zero-update preflight."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


RESET_TARGETS_VERSION = "thunder_cts_v13_reset_targets_v1"
NONZERO_EPS = 0.05
RAMP_STEPS = 10
RAMP_SECONDS = 0.2


def timer_resampled(current: np.ndarray, previous: np.ndarray, dt: float) -> np.ndarray:
    return current > previous + dt * 0.5


def expected_startup_command(
    goal: torch.Tensor, ramp: torch.Tensor, age_steps: torch.Tensor, step_dt: float = 0.02
) -> torch.Tensor:
    factor = torch.where(
        ramp,
        (age_steps.to(goal.dtype) * step_dt / RAMP_SECONDS).clamp(0.0, 1.0),
        torch.ones_like(age_steps, dtype=goal.dtype),
    )
    return goal * factor[:, None]


def reset_count_payload(counts: torch.Tensor) -> dict[str, Any]:
    if counts.shape != (2, 10, 2) or counts.dtype != torch.long:
        raise RuntimeError(f"Expected long startup_reset_counts[2,10,2], got {counts.dtype} {tuple(counts.shape)}")
    values = counts.detach().cpu().tolist()
    return {
        "axes": ["learner", "kind", "mode"],
        "learner_labels": ["teacher", "student"],
        "mode_labels": ["direct", "ramp"],
        "counts": values,
        "all_cells_positive": all(value > 0 for learner in values for kind in learner for value in kind),
    }


class Stats:
    """Validate V13 startup commands while preserving the real training path."""

    def __init__(self, env: Any, alg: Any, output: Path):
        from startup_repair import learner_cohort_mask

        self.env, self.alg, self.output = env, alg, output
        self.step = 0
        self.command = env.command_manager.get_term("base_velocity")
        self.robot = env.scene["robot"]
        self.learner = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.learner[alg.student_env_idxs] = 1
        self.normal_cohort = learner_cohort_mask(self.learner)
        self.episode = np.zeros(env.num_envs, dtype=np.int64)
        self.max_age = defaultdict(float)
        self.timeout_ends = Counter()
        self.native_intervals = defaultdict(list)
        self.native_zero_translation_intervals = Counter()
        self.previous_time_left = self.command.time_left.detach().cpu().numpy().copy()
        self.last_native_resample_age = np.full(env.num_envs, np.nan)
        self.last_native_translation_nonzero = np.zeros(env.num_envs, dtype=bool)
        self.startup_observations = Counter()
        self.step_zero = Counter()
        self.ramp_checks = Counter()
        self.violations: list[str] = []
        self.pending: dict[str, Any] | None = None
        self.term_snapshot: dict[str, np.ndarray] | None = None
        self._validate_configuration()

        receipt = getattr(env, "_thunder_reset_targets_installation", None)
        if receipt != {"version": RESET_TARGETS_VERSION, "robot": "robot"}:
            raise RuntimeError(f"Missing V13 reset target receipt: {receipt!r}")
        self.reset_target_receipt = dict(receipt)
        error = (self.robot.data.joint_pos_target - self.robot.data.joint_pos).abs().max()
        self.first_learn_target_actual_max_abs_error = float(error.item())
        if self.first_learn_target_actual_max_abs_error > 1e-6:
            raise RuntimeError("First learn target does not equal the actual reset joint pose")

        manager = env.termination_manager
        self._original_termination_compute = manager.compute
        self.termination_timeouts = {
            name: bool(manager.get_term_cfg(name).time_out) for name in manager.active_terms
        }

        def capture_termination(*args: Any, **kwargs: Any) -> Any:
            result = self._original_termination_compute(*args, **kwargs)
            self.term_snapshot = {
                name: manager.get_term(name).detach().cpu().numpy().copy()
                for name in manager.active_terms
            }
            return result

        manager.compute = capture_termination

    def _validate_configuration(self) -> None:
        command = self.command
        required = {
            "startup_goal": ((self.env.num_envs, 3), torch.float32),
            "startup_kind": ((self.env.num_envs,), torch.long),
            "startup_active": ((self.env.num_envs,), torch.bool),
            "startup_ramp": ((self.env.num_envs,), torch.bool),
            "startup_selected": ((self.env.num_envs,), torch.bool),
            "startup_started_step": ((self.env.num_envs,), torch.long),
        }
        for name, (shape, dtype) in required.items():
            value = getattr(command, name, None)
            if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype:
                actual = None if value is None else (tuple(value.shape), value.dtype)
                raise RuntimeError(f"Invalid {name}: {actual}; expected {(shape, dtype)}")
        reset_count_payload(command.startup_reset_counts)
        episode_seconds = float(self.env.max_episode_length * self.env.step_dt)
        resampling_seconds = float(command.cfg.resampling_time)
        if not np.isclose(episode_seconds, 25.0, atol=1e-6):
            raise RuntimeError(f"Expected 25 second episodes, got {episode_seconds}")
        if not np.isclose(resampling_seconds, 5.0, atol=1e-6):
            raise RuntimeError(f"Expected 5 second native resampling, got {resampling_seconds}")
        if "startup_window" in self.env.termination_manager.active_terms:
            raise RuntimeError("V13 must not install startup_window termination")
        if (command.startup_kind[~command.startup_selected] != -1).any():
            raise RuntimeError("Unselected startup_kind must be -1")

    def close(self) -> None:
        self.env.termination_manager.compute = self._original_termination_compute

    @staticmethod
    def _add_bins(
        counter: Counter,
        mask: torch.Tensor,
        learner: torch.Tensor,
        kind: torch.Tensor,
        ramp: torch.Tensor,
    ) -> None:
        if not mask.any():
            return
        bins = learner[mask] * 20 + kind[mask] * 2 + ramp[mask].long()
        counts = torch.bincount(bins, minlength=40).detach().cpu().tolist()
        for index, count in enumerate(counts):
            if count:
                counter[(index // 20, (index % 20) // 2, index % 2)] += count

    @torch.no_grad()
    def before(self, _obs: Any, _actions: torch.Tensor) -> None:
        command = self.command.command
        goal = self.command.startup_goal
        kind = self.command.startup_kind
        active = self.command.startup_active
        ramp = self.command.startup_ramp
        selected = self.command.startup_selected
        episode_age_steps = self.env.episode_length_buf
        startup_age_steps = int(self.env.common_step_counter) - self.command.startup_started_step
        flat = self.command.terrain_idxs == self.command.terrain_type2idx["flat"]
        normal_flat = flat & self.normal_cohort
        learner = self.learner

        if (active & (~selected | ~flat)).any():
            self.violations.append(f"step {self.step}: active startup escaped selected flat cohort")
        if ((kind[active] < 0) | (kind[active] > 9)).any():
            self.violations.append(f"step {self.step}: active kind outside 0..9")

        early = active & selected & (startup_age_steps >= 0) & (startup_age_steps <= RAMP_STEPS)
        if early.any():
            expected = expected_startup_command(
                goal[early], ramp[early], startup_age_steps[early], float(self.env.step_dt)
            )
            error = (command[early] - expected).abs().amax(dim=1)
            if (error > 1e-5).any():
                self.violations.append(f"step {self.step}: startup command trajectory mismatch")
            self._add_bins(self.startup_observations, early, learner, kind, ramp)
            step_zero = early & (startup_age_steps == 0)
            self._add_bins(self.step_zero, step_zero, learner, kind, ramp)
            magnitude = command.abs().amax(dim=1)
            if (step_zero & ramp & (magnitude > 1e-6)).any():
                self.violations.append(f"step {self.step}: ramp step zero was nonzero")
            if (step_zero & ~ramp & (magnitude <= NONZERO_EPS)).any():
                self.violations.append(f"step {self.step}: direct step zero was zero")
            self._add_bins(self.ramp_checks, early & ramp, learner, kind, ramp)

        age = (episode_age_steps * self.env.step_dt).detach().cpu().numpy()
        learner_np = learner.detach().cpu().numpy()
        normal_flat_np = normal_flat.detach().cpu().numpy()
        for group in (0, 1):
            members = normal_flat_np & (learner_np == group)
            if members.any():
                self.max_age[group] = max(self.max_age[group], float(age[members].max()))

        time_left = self.command.time_left.detach().cpu().numpy().copy()
        timer_event = timer_resampled(time_left, self.previous_time_left, self.env.step_dt)
        native = normal_flat_np & (~active).detach().cpu().numpy() & (age >= 5.0 - self.env.step_dt)
        translating = (command[:, :2].abs().amax(dim=1) > NONZERO_EPS).detach().cpu().numpy()
        for eid in np.flatnonzero(timer_event & native):
            previous = self.last_native_resample_age[eid]
            if np.isfinite(previous):
                interval = float(age[eid] - previous)
                group = int(learner_np[eid])
                if self.last_native_translation_nonzero[eid] and translating[eid]:
                    self.native_intervals[group].append(interval)
                elif interval < 4.9:
                    self.native_zero_translation_intervals[group] += 1
            self.last_native_resample_age[eid] = age[eid]
            self.last_native_translation_nonzero[eid] = translating[eid]
        self.previous_time_left = time_left
        self.pending = {"age": age, "learner": learner_np, "normal_flat": normal_flat_np}
        self.term_snapshot = None

    def after(self, _obs: Any, _rewards: torch.Tensor, dones: torch.Tensor, _extras: Any) -> None:
        if self.pending is None or self.term_snapshot is None:
            raise RuntimeError("Missing paired pre-step state or termination snapshot")
        done = dones.detach().cpu().numpy().astype(bool)
        for eid in np.flatnonzero(done & self.pending["normal_flat"]):
            terms = [name for name, values in self.term_snapshot.items() if values[eid]]
            if self.pending["age"][eid] >= 24.9 and any(self.termination_timeouts[name] for name in terms):
                self.timeout_ends[int(self.pending["learner"][eid])] += 1
        self.last_native_resample_age[done] = np.nan
        self.last_native_translation_nonzero[done] = False
        self.episode += done
        self.step += 1
        self.pending = None

    def finish(self) -> dict[str, Any]:
        if self.pending is not None:
            raise RuntimeError("Unpaired before/after callback")
        self.close()
        counts = reset_count_payload(self.command.startup_reset_counts)
        native = {}
        for group, label in ((0, "teacher"), (1, "student")):
            intervals = self.native_intervals[group]
            native[label] = {
                "count": len(intervals),
                "minimum": min(intervals) if intervals else None,
                "maximum": max(intervals) if intervals else None,
                "mean": float(np.mean(intervals)) if intervals else None,
                "short_intervals_involving_zero_translation": self.native_zero_translation_intervals[group],
            }
        summary = {
            "steps": self.step,
            "simulated_seconds": self.step * float(self.env.step_dt),
            "num_envs": self.env.num_envs,
            "optimizer_steps": 0,
            "configuration": {
                "episode_seconds": 25.0,
                "native_resampling_seconds": 5.0,
                "evidence_scope": "flat normal cohort",
            },
            "maximum_episode_age_seconds": {
                "teacher": self.max_age[0], "student": self.max_age[1]
            },
            "normal_timeouts_age_ge_24p9_seconds": {
                "teacher": self.timeout_ends[0], "student": self.timeout_ends[1]
            },
            "native_resample_intervals_seconds": native,
            "algorithm_startup_anchor_counts": dict(self.alg.anchor_sample_counts),
            "startup_anchor": self.alg.anchor_state(),
        }
        v13 = {
            "reset_target_receipt": self.reset_target_receipt,
            "first_learn_target_actual_max_abs_error": self.first_learn_target_actual_max_abs_error,
            "startup_reset_counts": counts,
            "startup_observations": {str(key): value for key, value in sorted(self.startup_observations.items())},
            "step_zero_observations": {str(key): value for key, value in sorted(self.step_zero.items())},
            "ramp_first_ten_step_checks": {str(key): value for key, value in sorted(self.ramp_checks.items())},
            "violations": self.violations,
            "counter_scope": "production clears after learner assignment and counts the following real reset",
        }
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "full_episode_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.output / "v13_summary.json").write_text(json.dumps(v13, indent=2), encoding="utf-8")
        return summary
