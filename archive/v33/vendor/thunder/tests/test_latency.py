from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "source/doso_train/doso_train/tasks/terrain_inward/latency.py"
SPEC = importlib.util.spec_from_file_location("doso_latency", MODULE_PATH)
latency = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(latency)


class FakeDelayBuffer:
    def __init__(self, history_length: int, batch_size: int, device: str):
        self.history_length = history_length
        self.batch_size = batch_size
        self.device = device
        self.time_lags = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._history: list[torch.Tensor] = []
        self._ages = torch.zeros(batch_size, dtype=torch.long, device=device)

    def set_time_lag(self, lag, batch_ids=None):
        if batch_ids is None:
            self.time_lags[:] = lag
        else:
            self.time_lags[batch_ids] = lag

    def reset(self, batch_ids=None):
        ids = slice(None) if batch_ids is None else batch_ids
        self._ages[ids] = 0

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        self._history.append(data.clone())
        output = data.clone()
        for env_id in range(self.batch_size):
            lag = int(self.time_lags[env_id])
            if self._ages[env_id] >= lag:
                output[env_id] = self._history[-1 - lag][env_id]
        self._ages += 1
        return output


def make_processor(**kwargs):
    return latency.ThunderLatency(2, "cpu", delay_buffer_cls=FakeDelayBuffer, **kwargs)


def policy_obs(value: float) -> torch.Tensor:
    obs = torch.full((2, 53), value)
    obs[:, 6:9] = value + 100.0
    obs[:, 37:53] = value + 200.0
    return obs


def test_observation_delay_only_changes_sensor_slices_and_uses_raw_last_action():
    processor = make_processor(obs_delay_min_frames=1, obs_delay_max_frames=1)
    first = policy_obs(1.0)
    processor.reset(first)
    processor.process_observation(first)
    raw_action = torch.full((2, 16), 7.0)
    processor.process_action(raw_action)

    current = policy_obs(2.0)
    output = processor.process_observation(current)

    assert torch.all(output[:, 0:6] == 1.0)
    assert torch.all(output[:, 9:37] == 1.0)
    assert torch.all(output[:, 6:9] == 102.0)
    assert torch.all(output[:, 37:53] == 7.0)
    assert output.shape == (2, 53)


def test_observation_dropout_holds_last_sensor_frame_but_not_command():
    processor = make_processor(obs_drop_start_prob=1.0, obs_drop_max_frames=1)
    first = policy_obs(1.0)
    processor.reset(first)
    output = processor.process_observation(policy_obs(2.0))
    recovered = processor.process_observation(policy_obs(3.0))

    assert torch.all(output[:, 0:6] == 1.0)
    assert torch.all(output[:, 9:37] == 1.0)
    assert torch.all(output[:, 6:9] == 102.0)
    assert torch.all(recovered[:, 0:6] == 3.0)


def test_action_delay_zero_fills_reset_then_replays_fifo():
    processor = make_processor(action_delay_min_frames=2, action_delay_max_frames=2)
    processor.reset(policy_obs(0.0))
    action_1 = torch.full((2, 16), 1.0)
    action_2 = torch.full((2, 16), 2.0)
    action_3 = torch.full((2, 16), 3.0)

    assert torch.count_nonzero(processor.process_action(action_1)) == 0
    assert torch.count_nonzero(processor.process_action(action_2)) == 0
    assert torch.equal(processor.process_action(action_3), action_1)


def test_episode_reset_samples_bounded_per_env_delays_and_clears_last_action():
    processor = make_processor(
        obs_delay_min_frames=0,
        obs_delay_max_frames=3,
        action_delay_min_frames=0,
        action_delay_max_frames=3,
        seed=42,
    )
    obs = policy_obs(0.0)
    processor.reset(obs)
    processor.process_action(torch.ones((2, 16)))
    processor.reset(obs, torch.tensor([1]))
    output = processor.process_observation(obs)

    assert processor.observation_lags.min() >= 0 and processor.observation_lags.max() <= 3
    assert processor.action_lags.min() >= 0 and processor.action_lags.max() <= 3
    assert torch.all(output[1, 37:53] == 0.0)
    assert torch.all(output[0, 37:53] == 1.0)


class FakeVectorEnv(gym.Env):
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.last_applied_action = None
        self.critic = torch.full((2, 56), 9.0)

    def reset(self, **kwargs):
        return {"policy": policy_obs(1.0), "critic": self.critic}, {"reset": True}

    def step(self, action):
        self.last_applied_action = action
        obs = {"policy": policy_obs(2.0), "critic": self.critic}
        return obs, torch.zeros(2), torch.tensor([False, True]), torch.zeros(2, dtype=torch.bool), {}


def test_wrapper_leaves_critic_unchanged_and_resets_done_episode_before_observation():
    env = FakeVectorEnv()
    wrapper = latency.ThunderLatencyWrapper(
        env,
        action_delay_min_frames=1,
        action_delay_max_frames=1,
        delay_buffer_cls=FakeDelayBuffer,
    )
    obs, _ = wrapper.reset()
    critic = obs["critic"]
    next_obs, *_ = wrapper.step(torch.ones((2, 16)))

    assert next_obs["critic"] is critic
    assert torch.count_nonzero(env.last_applied_action) == 0
    assert torch.all(next_obs["policy"][1, 37:53] == 0.0)
    assert torch.all(next_obs["policy"][0, 37:53] == 1.0)
