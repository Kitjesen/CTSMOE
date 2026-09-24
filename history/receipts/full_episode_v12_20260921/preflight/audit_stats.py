"""Streaming statistics for the 60 second full-episode zero-update preflight."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


NONZERO_EPS = 0.05
TRACE_STRIDE = 25  # 0.5 simulated seconds at 50 Hz.


def command_nonzero(command: np.ndarray) -> np.ndarray:
    return np.max(np.abs(command), axis=1) > NONZERO_EPS


def command_changed(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return np.max(np.abs(current - previous), axis=1) > NONZERO_EPS


def translation_nonzero(command: np.ndarray) -> np.ndarray:
    """Yaw-only commands belong to the zero-translation short-segment exception."""
    return np.max(np.abs(command[:, :2]), axis=1) > NONZERO_EPS


def timer_resampled(current: np.ndarray, previous: np.ndarray, dt: float) -> np.ndarray:
    """Detect actual sampler events even when the sampled command value repeats."""
    return current > previous + dt * 0.5


class CommandClock:
    """Track real nonzero and unchanged-command ages without touching RNG state."""

    def __init__(self, count: int, dt: float):
        self.dt = float(dt)
        self.previous = np.zeros((count, 3), dtype=np.float64)
        self.previous_episode = np.full(count, -1, dtype=np.int64)
        self.nonzero_age = np.zeros(count, dtype=np.float64)
        self.segment_age = np.zeros(count, dtype=np.float64)

    def update(
        self, command: np.ndarray, episode: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nonzero = command_nonzero(command)
        reset = episode != self.previous_episode
        changed = command_changed(command, self.previous) | reset
        self.nonzero_age = np.where(nonzero & ~reset, self.nonzero_age + self.dt, 0.0)
        self.segment_age = np.where(nonzero & ~changed, self.segment_age + self.dt, 0.0)
        self.previous = command.copy()
        self.previous_episode = episode.copy()
        return self.nonzero_age.copy(), self.segment_age.copy(), changed, reset


class VectorMoments:
    """Online per-column moments, avoiding trajectory-sized observation storage."""

    def __init__(self, width: int):
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.square = np.zeros(width, dtype=np.float64)
        self.abs_max = np.zeros(width, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        if not len(values):
            return
        values = np.asarray(values, dtype=np.float64)
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.square += np.square(values).sum(axis=0)
        self.abs_max = np.maximum(self.abs_max, np.abs(values).max(axis=0))

    def summary(self, names: list[str]) -> dict[str, Any]:
        if not self.count:
            return {"states": 0, "joints": {}}
        mean = self.total / self.count
        rms = np.sqrt(self.square / self.count)
        return {
            "states": self.count,
            "joints": {
                name: {"mean": float(mean[i]), "rms": float(rms[i]), "abs_max": float(self.abs_max[i])}
                for i, name in enumerate(names)
            },
        }


class Stats:
    """Aggregate flat special/ordinary behavior and true pre-reset terminal causes."""

    def __init__(self, env: Any, alg: Any, output: Path):
        from anchor_objective import normal_start_mask
        from startup_repair import learner_cohort_mask

        self.env, self.alg, self.output = env, alg, output
        self.mask_fn = normal_start_mask
        self.step = 0
        self.episode = np.zeros(env.num_envs, dtype=np.int64)
        self.learner = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.learner[alg.student_env_idxs] = 1
        self.cohort = learner_cohort_mask(self.learner)
        torch.testing.assert_close(self.cohort, alg.anchor_cohort)
        self.clock = CommandClock(env.num_envs, env.step_dt)
        self.robot = env.scene["robot"]
        self.command = env.command_manager.get_term("base_velocity")
        self.request_term = env.action_manager.get_term("joint_pos")
        self.request_ids = self.request_term.joint_indices
        self.joint_names = [self.robot.joint_names[i] for i in self.request_ids]
        width = len(self.request_ids)
        self.pose = defaultdict(lambda: {name: VectorMoments(width) for name in ("actual", "target", "request")})
        self.states = Counter()
        self.episodes = defaultdict(set)
        self.max_age = defaultdict(float)
        self.duration = Counter()
        self.anchor = Counter()
        self.ends = Counter()
        self.physical_by_term = Counter()
        self.normal_timeouts_24p9 = Counter()
        self.resample = defaultdict(list)
        self.resample_generations = Counter()
        self.short_zero_intervals = Counter()
        self.previous_time_left = self.command.time_left.detach().cpu().numpy().copy()
        self.last_resample_age = np.zeros(env.num_envs, dtype=np.float64)
        initial_command = self.command.command.detach().cpu().numpy()
        self.last_resample_translation_nonzero = translation_nonzero(initial_command)
        self.trace_ids: dict[tuple[int, bool], int] = {}
        self.traces: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self.term_snapshot: dict[str, np.ndarray] | None = None

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
        self.configuration = self._configuration()

    def _configuration(self) -> dict[str, Any]:
        active_terms = list(self.env.termination_manager.active_terms)
        episode_seconds = float(self.env.max_episode_length * self.env.step_dt)
        resampling_seconds = float(self.command.cfg.resampling_time)
        if not np.isclose(episode_seconds, 25.0, atol=1e-6):
            raise RuntimeError(f"Expected 25 second episodes, got {episode_seconds}")
        if not np.isclose(resampling_seconds, 5.0, atol=1e-6):
            raise RuntimeError(f"Expected 5 second ordinary sampler, got {resampling_seconds}")
        if "startup_window" in active_terms:
            raise RuntimeError("Full-episode profile must not install startup_window termination")
        if type(self.command).__name__ != "MixedCommand":
            raise RuntimeError(f"Expected normal MixedCommand sampler, got {type(self.command).__name__}")
        return {
            "episode_seconds": episode_seconds,
            "ordinary_resampling_seconds": resampling_seconds,
            "startup_window": None,
            "sampler": "normal_training_sampler",
            "command_term_class": type(self.command).__name__,
            "termination_terms": active_terms,
            "step_dt": float(self.env.step_dt),
        }

    @staticmethod
    def _label(learner: int, special: bool) -> str:
        return f"{'student' if learner else 'teacher'}_{'special' if special else 'ordinary'}"

    def close(self) -> None:
        self.env.termination_manager.compute = self._original_termination_compute

    @torch.no_grad()
    def before(self, _obs: Any, _actions: torch.Tensor) -> None:
        rng_before = torch.cuda.get_rng_state().clone()
        command_t = self.command.command
        flat_t = self.command.terrain_idxs == self.command.terrain_type2idx["flat"]
        special_t = flat_t & self.cohort
        age_t = self.env.episode_length_buf * self.env.step_dt
        command = command_t.detach().cpu().numpy().copy()
        flat = flat_t.cpu().numpy()
        special = special_t.cpu().numpy()
        learner = self.learner.cpu().numpy()
        age = age_t.cpu().numpy()
        nonzero_age, segment_age, _changed, reset = self.clock.update(command, self.episode)
        translating = translation_nonzero(command)
        time_left = self.command.time_left.detach().cpu().numpy().copy()
        anchor = (self.mask_fn(flat_t, age_t, command_t) & self.cohort).cpu().numpy()

        data = self.robot.data
        ids = flat_t.nonzero().flatten()
        actual = (data.joint_pos[ids][:, self.request_ids] - data.default_joint_pos[ids][:, self.request_ids]).cpu().numpy()
        target = (data.joint_pos_target[ids][:, self.request_ids] - data.default_joint_pos[ids][:, self.request_ids]).cpu().numpy()
        request = (self.request_term.requested_actions[ids] - data.default_joint_pos[ids][:, self.request_ids]).cpu().numpy()
        for row, eid in enumerate(ids.cpu().tolist()):
            key = (int(learner[eid]), bool(special[eid]))
            self.states[key] += 1
            self.episodes[key].add((eid, int(self.episode[eid])))
            self.max_age[key] = max(self.max_age[key], float(age[eid]))
            self.anchor[(key, bool(anchor[eid]))] += 1
            self.duration[(key, "nonzero_ge_3s")] += int(nonzero_age[eid] >= 3.0)
            self.duration[(key, "nonzero_ge_5s")] += int(nonzero_age[eid] >= 5.0)
            self.duration[(key, "segment_ge_3s")] += int(segment_age[eid] >= 3.0)
            self.duration[(key, "segment_ge_5s")] += int(segment_age[eid] >= 5.0)
            self.pose[key]["actual"].add(actual[row : row + 1])
            self.pose[key]["target"].add(target[row : row + 1])
            self.pose[key]["request"].add(request[row : row + 1])
            if key not in self.trace_ids:
                self.trace_ids[key] = eid

        # MixedCommand's timer is the source of truth for resampling. Command-value
        # comparison would miss a repeated/clamped sample and report false 10/15s gaps.
        timer_increased = timer_resampled(time_left, self.previous_time_left, self.env.step_dt)
        for eid in np.flatnonzero(reset):
            self.last_resample_age[eid] = 0.0
            self.last_resample_translation_nonzero[eid] = translating[eid]
        for eid in np.flatnonzero(timer_increased & flat & ~reset):
            key = (int(learner[eid]), bool(special[eid]))
            interval = float(age[eid] - self.last_resample_age[eid])
            if self.last_resample_translation_nonzero[eid] and translating[eid]:
                self.resample[key].append(interval)
                generation = "episode_0" if self.episode[eid] == 0 else "episode_1_plus"
                self.resample_generations[(key, generation)] += 1
            else:
                # Zero XY translation remains the sampler's allowed short-segment
                # exception even when yaw is nonzero.
                self.short_zero_intervals[key] += int(interval < 4.9)
            self.last_resample_age[eid] = age[eid]
            self.last_resample_translation_nonzero[eid] = translating[eid]
        self.previous_time_left = time_left

        if self.step % TRACE_STRIDE == 0:
            flat_rows = {int(eid): row for row, eid in enumerate(ids.cpu().tolist())}
            for key, eid in self.trace_ids.items():
                if flat[eid]:
                    row = flat_rows[eid]
                    self.traces.append({
                        "step": self.step,
                        "env_id": eid,
                        "group": self._label(*key),
                        "episode": int(self.episode[eid]),
                        "age_seconds": float(age[eid]),
                        "command": command[eid].tolist(),
                        "nonzero_age_seconds": float(nonzero_age[eid]),
                        "segment_age_seconds": float(segment_age[eid]),
                        "actual_rms": float(np.sqrt(np.mean(np.square(actual[row])))),
                        "target_rms": float(np.sqrt(np.mean(np.square(target[row])))),
                        "request_rms": float(np.sqrt(np.mean(np.square(request[row])))),
                    })
        self.pending = {"flat": flat, "special": special, "learner": learner, "age": age}
        self.term_snapshot = None
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng_before)

    def after(self, _obs: Any, _rewards: torch.Tensor, dones: torch.Tensor, _extras: Any) -> None:
        if self.pending is None or self.term_snapshot is None:
            raise RuntimeError("Missing paired pre-step state or pre-auto-reset termination snapshot")
        done = dones.cpu().numpy().astype(bool)
        for eid in np.flatnonzero(done & self.pending["flat"]):
            key = (int(self.pending["learner"][eid]), bool(self.pending["special"][eid]))
            age = float(self.pending["age"][eid])
            terms = [name for name, values in self.term_snapshot.items() if values[eid]]
            self.ends[(key, "all")] += 1
            timeout_terms = [term for term in terms if self.termination_timeouts[term]]
            physical_terms = [term for term in terms if not self.termination_timeouts[term]]
            self.ends[(key, "timeout")] += int(bool(timeout_terms))
            self.ends[(key, "physical")] += int(bool(physical_terms))
            self.normal_timeouts_24p9[key] += int(age >= 24.9 and bool(timeout_terms))
            for term in physical_terms:
                self.physical_by_term[(key, term)] += 1
            if eid in self.trace_ids.values():
                self.traces.append({"step": self.step, "env_id": int(eid), "terminal": True, "terms": terms})
        self.episode += done
        self.step += 1
        self.pending = None

    def finish(self) -> dict[str, Any]:
        if self.pending is not None:
            raise RuntimeError("Unpaired before/after callback")
        self.close()
        groups = {}
        for key in sorted(self.states):
            label = self._label(*key)
            intervals = self.resample[key]
            groups[label] = {
                "states": self.states[key],
                "distinct_episodes": len(self.episodes[key]),
                "maximum_episode_age_seconds": self.max_age[key],
                "episode_ends": self.ends[(key, "all")],
                "timeout_ends": self.ends[(key, "timeout")],
                "normal_timeouts_age_ge_24p9_seconds": self.normal_timeouts_24p9[key],
                "physical_ends": self.ends[(key, "physical")],
                "physical_failures_by_term": {
                    term: count for (term_key, term), count in self.physical_by_term.items() if term_key == key
                },
                "duration_state_counts": {
                    metric: self.duration[(key, metric)]
                    for metric in ("nonzero_ge_3s", "nonzero_ge_5s", "segment_ge_3s", "segment_ge_5s")
                },
                "startup_anchor_counts": {
                    "false": self.anchor[(key, False)], "true": self.anchor[(key, True)]
                },
                "normal_translation_resample_intervals_seconds": {
                    "count": len(intervals),
                    "minimum": min(intervals) if intervals else None,
                    "maximum": max(intervals) if intervals else None,
                    "mean": float(np.mean(intervals)) if intervals else None,
                    "expected": 5.0,
                    "episode_generation_counts": {
                        generation: self.resample_generations[(key, generation)]
                        for generation in ("episode_0", "episode_1_plus")
                    },
                },
                "short_intervals_involving_zero_translation": self.short_zero_intervals[key],
                "pose_offset_from_default_rad": {
                    name: moments.summary(self.joint_names) for name, moments in self.pose[key].items()
                },
            }
        summary = {
            "steps": self.step,
            "simulated_seconds": self.step * float(self.env.step_dt),
            "num_envs": self.env.num_envs,
            "optimizer_steps": 0,
            "configuration": self.configuration,
            "groups": groups,
            "timing": "pre-step state; termination terms captured after compute and before auto-reset",
            "sampling": "original training noise and stochastic policy actions retained",
            "algorithm_startup_anchor_counts": dict(self.alg.anchor_sample_counts),
            "startup_anchor": self.alg.anchor_state(),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "full_episode_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.output / "representative_flat_traces.json").write_text(
            json.dumps({"stride_steps": TRACE_STRIDE, "traces": self.traces}, indent=2), encoding="utf-8"
        )
        return summary
