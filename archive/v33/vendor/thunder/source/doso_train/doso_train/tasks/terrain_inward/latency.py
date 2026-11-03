# Copyright (c) 2026 Shanghai Qiongpei Technology Co., Ltd.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Policy observation and action latency for Thunder sim-to-real training."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import gymnasium as gym
import torch

try:
    from isaaclab.utils.buffers import DelayBuffer as _IsaacDelayBuffer
except ModuleNotFoundError:  # Allows CPU-only unit tests to inject a small fake.
    _IsaacDelayBuffer = None


POLICY_OBS_DIM = 53
ACTION_DIM = 16
_SENSOR_SLICES = (slice(0, 3), slice(3, 6), slice(9, 21), slice(21, 37))
_LAST_ACTION_SLICE = slice(37, 53)


class ThunderLatency:
    """Apply per-environment latency without changing the policy interface."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        obs_delay_min_frames: int = 0,
        obs_delay_max_frames: int = 0,
        obs_drop_start_prob: float = 0.0,
        obs_drop_max_frames: int = 0,
        action_delay_min_frames: int = 0,
        action_delay_max_frames: int = 0,
        seed: int | None = None,
        delay_buffer_cls: Callable[..., object] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.obs_delay_range = self._validate_range(
            "observation delay", obs_delay_min_frames, obs_delay_max_frames
        )
        self.action_delay_range = self._validate_range(
            "action delay", action_delay_min_frames, action_delay_max_frames
        )
        if not 0.0 <= obs_drop_start_prob <= 1.0:
            raise ValueError("obs_drop_start_prob must be in [0, 1]")
        if obs_drop_max_frames < 0:
            raise ValueError("obs_drop_max_frames must be non-negative")
        self.obs_drop_start_prob = obs_drop_start_prob
        self.obs_drop_max_frames = obs_drop_max_frames

        buffer_cls = delay_buffer_cls or _IsaacDelayBuffer
        if buffer_cls is None:
            raise RuntimeError("Isaac Lab is unavailable; pass delay_buffer_cls for tests")
        self._obs_buffer = buffer_cls(self.obs_delay_range[1], num_envs, str(self.device))
        self._action_buffer = buffer_cls(self.action_delay_range[1], num_envs, str(self.device))
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device).manual_seed(seed)

        sensor_dim = sum(part.stop - part.start for part in _SENSOR_SLICES)
        self._last_valid_sensors = torch.zeros((num_envs, sensor_dim), device=self.device)
        self._previous_raw_action = torch.zeros((num_envs, ACTION_DIM), device=self.device)
        self._drop_remaining = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._drop_cooldown = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._action_age = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._obs_lags = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._action_lags = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @property
    def observation_lags(self) -> torch.Tensor:
        return self._obs_lags.clone()

    @property
    def action_lags(self) -> torch.Tensor:
        return self._action_lags.clone()

    def reset(
        self, policy_obs: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Reset selected episodes and sample their fixed observation/action delays."""
        self._validate_policy_obs(policy_obs)
        ids = self._env_ids(env_ids)
        self._obs_buffer.reset(ids)
        self._action_buffer.reset(ids)
        self._obs_lags[ids] = self._sample_delay(self.obs_delay_range, len(ids))
        self._action_lags[ids] = self._sample_delay(self.action_delay_range, len(ids))
        self._obs_buffer.set_time_lag(self._obs_lags[ids], ids)
        self._action_buffer.set_time_lag(self._action_lags[ids], ids)
        self._last_valid_sensors[ids] = self._sensors(policy_obs)[ids]
        self._previous_raw_action[ids] = 0.0
        self._drop_remaining[ids] = 0
        self._drop_cooldown[ids] = False
        self._action_age[ids] = 0

    def process_observation(self, policy_obs: torch.Tensor) -> torch.Tensor:
        """Delay sensor fields, hold dropped frames, and preserve current commands."""
        self._validate_policy_obs(policy_obs)
        delayed = self._obs_buffer.compute(self._sensors(policy_obs))
        holding = self._drop_remaining > 0
        if self.obs_drop_start_prob > 0.0 and self.obs_drop_max_frames > 0:
            starts = (~holding) & (~self._drop_cooldown) & (
                torch.rand(self.num_envs, device=self.device, generator=self._generator)
                < self.obs_drop_start_prob
            )
            if starts.any():
                durations = torch.randint(
                    1,
                    self.obs_drop_max_frames + 1,
                    (int(starts.sum().item()),),
                    device=self.device,
                    generator=self._generator,
                )
                self._drop_remaining[starts] = durations
                holding |= starts

        output_sensors = delayed.clone()
        output_sensors[holding] = self._last_valid_sensors[holding]
        self._last_valid_sensors[~holding] = delayed[~holding]
        self._drop_remaining[holding] -= 1
        self._drop_cooldown.copy_(holding & (self._drop_remaining == 0))

        output = policy_obs.clone()
        self._write_sensors(output, output_sensors)
        output[:, _LAST_ACTION_SLICE] = self._previous_raw_action
        return output

    def process_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Return the delayed actuator action while remembering the raw policy output."""
        if raw_action.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"expected action shape {(self.num_envs, ACTION_DIM)}, got {tuple(raw_action.shape)}"
            )
        delayed = self._action_buffer.compute(raw_action)
        delayed[self._action_age < self._action_lags] = 0.0
        self._action_age += 1
        self._previous_raw_action.copy_(raw_action)
        return delayed

    @staticmethod
    def _validate_range(name: str, minimum: int, maximum: int) -> tuple[int, int]:
        if minimum < 0 or maximum < minimum:
            raise ValueError(f"invalid {name} range: ({minimum}, {maximum})")
        return minimum, maximum

    def _validate_policy_obs(self, policy_obs: torch.Tensor) -> None:
        if policy_obs.shape != (self.num_envs, POLICY_OBS_DIM):
            raise ValueError(
                f"expected policy observation shape {(self.num_envs, POLICY_OBS_DIM)}, "
                f"got {tuple(policy_obs.shape)}"
            )

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

    def _sample_delay(self, limits: tuple[int, int], count: int) -> torch.Tensor:
        return torch.randint(
            limits[0],
            limits[1] + 1,
            (count,),
            dtype=torch.long,
            device=self.device,
            generator=self._generator,
        )

    @staticmethod
    def _sensors(policy_obs: torch.Tensor) -> torch.Tensor:
        return torch.cat([policy_obs[:, part] for part in _SENSOR_SLICES], dim=-1)

    @staticmethod
    def _write_sensors(policy_obs: torch.Tensor, sensors: torch.Tensor) -> None:
        offset = 0
        for part in _SENSOR_SLICES:
            width = part.stop - part.start
            policy_obs[:, part] = sensors[:, offset : offset + width]
            offset += width


class ThunderLatencyWrapper(gym.Wrapper):
    """Gym vector wrapper that applies :class:`ThunderLatency` to policy I/O only."""

    def __init__(self, env: gym.Env, **latency_kwargs) -> None:
        super().__init__(env)
        self.latency = ThunderLatency(env.num_envs, env.device, **latency_kwargs)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        self.latency.reset(obs["policy"])
        obs["policy"] = self.latency.process_observation(obs["policy"])
        return (obs, *result[1:]) if isinstance(result, tuple) else obs

    def step(self, action):
        result = self.env.step(self.latency.process_action(action))
        obs, terminated, truncated = result[0], result[2], result[3]
        done_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
        if len(done_ids):
            self.latency.reset(obs["policy"], done_ids)
        obs["policy"] = self.latency.process_observation(obs["policy"])
        return (obs, *result[1:])
