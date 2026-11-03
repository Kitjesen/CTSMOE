"""Stateless NEWPOSE deployment: frame-major history530 -> actions16."""

import copy
from pathlib import Path

import torch


class ThunderHistoryPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        if (policy.num_actor_obs, policy.num_single_obs, policy.num_actions) != (530, 53, 16):
            raise ValueError("Unsupported Thunder observation/action shape")
        if policy.state_dependent_std:
            raise ValueError("The frozen NEWPOSE policy does not use state-dependent std")
        self.encoder = copy.deepcopy(policy.student_moe_encoder).cpu().eval()
        self.actor = copy.deepcopy(policy.actor).cpu().eval()
        self.history_normalizer = copy.deepcopy(policy.actor_obs_normalizer).cpu().eval()
        self.frame_normalizer = copy.deepcopy(policy.single_obs_normalizer).cpu().eval()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 2 or history.shape[-1] != 530:
            raise ValueError("Expected batch x 530 frame-major history, current frame last")
        latent, _ = self.encoder(self.history_normalizer(history))
        return self.actor(torch.cat((latent, self.frame_normalizer(history[:, -53:])), dim=-1))


def export_jit(policy, path, filename="policy.pt"):
    destination = Path(path) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    module = ThunderHistoryPolicy(policy).eval()
    torch.jit.script(module).save(str(destination))
    return destination


def export_onnx(policy, path, filename="policy.onnx", verbose=False):
    # Reuse the upstream serializer, supplying Thunder's forward/extraction.
    from rsl_rl.moe_cts.exporter import OnnxMoeCtsExporter
    module = ThunderHistoryPolicy(policy).eval()
    module.num_actor_obs = 530
    module.verbose = verbose
    OnnxMoeCtsExporter.export(module, str(path), filename)
    return Path(path) / filename
