"""The old NEWPOSE history contract; no policy/algorithm reimplementation."""

import torch
from tensordict import TensorDict


class FrameHistory:
    """Ten complete frames, oldest first, including the current frame.

    A reset repeats the first post-reset frame ten times, exactly as the old
    NEWPOSE History did. Call advance once per control step, never from a getter.
    """

    def __init__(self, frame):
        if frame.ndim != 2 or frame.shape[-1] != 53:
            raise ValueError("Thunder requires an (environment, 53) frame")
        self.data = frame[:, None, :].repeat(1, 10, 1)

    def advance(self, frame, done):
        if frame.shape != self.data[:, -1].shape or done.numel() != len(frame):
            raise ValueError("History environment/frame shape changed")
        self.data = torch.cat((self.data[:, 1:], frame[:, None, :]), dim=1)
        reset = done.reshape(-1).bool()
        self.data[reset] = frame[reset, None, :]

    def pack(self, critic):
        if critic.ndim != 2 or len(critic) != len(self.data):
            raise ValueError("Critic and student environment batches differ")
        return TensorDict({"policy": self.data.flatten(1), "single_obs": self.data[:, -1],
                           "critic": critic}, batch_size=[len(critic)])


def validate_observations(obs, critic_dim=309):
    expected = {"policy": 530, "single_obs": 53, "critic": critic_dim}
    for key, dimension in expected.items():
        if key not in obs or obs[key].ndim != 2 or obs[key].shape[-1] != dimension:
            raise ValueError(f"Invalid Thunder {key} observation; expected batch x {dimension}")
        if not torch.isfinite(obs[key]).all():
            raise ValueError(f"Nonfinite {key} observation")
    if not torch.equal(obs["policy"][:, -53:], obs["single_obs"]):
        raise ValueError("History must contain the same current frame as single_obs")
