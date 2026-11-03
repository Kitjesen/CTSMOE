"""Synthetic runner-level timeout regression; never a formal training update."""

import argparse
import json
import math
from pathlib import Path

import torch

from .runner import ThunderMoeCtsRunner
from .timeout import terminal_timeout_reward
from .verify_bridge import SyntheticEnv, SyntheticRunner, runner_config


class TimeoutEnv(SyntheticEnv):
    def __init__(self):
        super().__init__()
        self.max_episode_length = 2
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.raw_rewards = None
        self.raw_extras = None

    def step(self, actions):
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.max_episode_length
        # Environment 2 simultaneously reaches a true terminal state and timeout.
        self.reset_terminated = done & (torch.arange(self.num_envs) == 2)
        final_frame = torch.randn(8, 53) * .1
        final_frame[:, -16:] = actions
        final_critic = torch.randn(8, 309) * .1
        frame, self.critic = final_frame.clone(), final_critic.clone()
        frame[done] = torch.randn(int(done.sum()), 53) * .1
        self.critic[done] = torch.randn(int(done.sum()), 309) * .1
        self.episode_length_buf[done] = 0
        self.history.advance(frame, done)
        self.raw_rewards = -.01 * actions.square().mean(-1)
        self.raw_extras = {"time_outs": done, "final_obs": {
            "single_obs": final_frame, "critic": final_critic}}
        return self.get_observations(), self.raw_rewards, done, self.raw_extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    env = TimeoutEnv()
    runner = SyntheticRunner(env, runner_config(), device="cpu")
    runner.load(args.source)
    upstream_process = runner.alg.process_env_step
    checked_steps = []

    def check_process(obs, rewards, dones, extras):
        expected, _, count = terminal_timeout_reward(
            runner.alg, env.raw_rewards, dones, env.raw_extras, env.reset_terminated)
        torch.testing.assert_close(rewards, expected, atol=0., rtol=0.)
        assert "time_outs" not in extras
        checked_steps.append(count)
        return upstream_process(obs, rewards, dones, extras)

    runner.alg.process_env_step = check_process
    losses = []
    runner.logger.log = lambda **kwargs: losses.append(kwargs["loss_dict"])
    runner.learn(1)
    assert checked_steps == [0, 7, 0, 7]
    assert runner.terminal_bootstraps == 14
    assert runner.completed_updates == 19539
    assert len(losses) == 1 and all(math.isfinite(v) for v in losses[0].values())
    result = {"scope": "synthetic CPU runner, no simulator, no persisted checkpoint, no formal progress",
              "upstream_algorithm": type(runner.alg).__module__, "steps": len(checked_steps),
              "terminal_bootstraps": runner.terminal_bootstraps,
              "simultaneous_true_terminations_excluded": 2,
              "corrected_rewards_exact": True, "upstream_duplicate_bootstrap_disabled": True,
              "losses": losses[0], "formal_updates_unchanged": 19538}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
