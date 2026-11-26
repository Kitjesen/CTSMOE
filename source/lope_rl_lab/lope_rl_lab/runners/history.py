"""Ten frame-major Lope observations, current frame last."""
import torch
from tensordict import TensorDict

from lope_rl_lab.tasks.locomotion.config import CRITIC_DIM, FRAME_DIM, HISTORY_LENGTH, POLICY_DIM


class FrameHistory:
    def __init__(self, frame):
        if frame.ndim != 2 or frame.shape[-1] != FRAME_DIM:
            raise ValueError("Lope requires an (environment, 45) frame")
        self.data = frame[:, None, :].repeat(1, HISTORY_LENGTH, 1)

    def advance(self, frame, done):
        if frame.shape != self.data[:, -1].shape or done.numel() != len(frame):
            raise ValueError("History environment/frame shape changed")
        self.data = torch.cat((self.data[:, 1:], frame[:, None, :]), dim=1)
        reset = done.reshape(-1).bool()
        self.data[reset] = frame[reset, None, :]

    def pack(self, critic):
        if critic.ndim != 2 or len(critic) != len(self.data) or critic.shape[-1] != CRITIC_DIM:
            raise ValueError("Critic and student environment batches differ")
        return TensorDict(
            {"policy": self.data.flatten(1), "single_obs": self.data[:, -1], "critic": critic},
            batch_size=[len(critic)],
        )


def validate_observations(obs, critic_dim=CRITIC_DIM):
    expected = {"policy": POLICY_DIM, "single_obs": FRAME_DIM, "critic": critic_dim}
    for key, dimension in expected.items():
        if key not in obs or obs[key].ndim != 2 or obs[key].shape[-1] != dimension:
            raise ValueError(f"Invalid Lope {key} observation; expected batch x {dimension}")
        if not torch.isfinite(obs[key]).all():
            raise ValueError(f"Nonfinite {key} observation")
    if not torch.equal(obs["policy"][:, -FRAME_DIM:], obs["single_obs"]):
        raise ValueError("History must contain the same current frame as single_obs")
