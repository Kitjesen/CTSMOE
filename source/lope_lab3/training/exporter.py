"""Deployment graph for a 450-value Lope history and 12 position actions."""
import copy
from pathlib import Path

import torch

from lope_lab3.spec import ACTION_DIM, FRAME_DIM, POLICY_DIM


class LopeHistoryPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        if (policy.num_actor_obs, policy.num_single_obs, policy.num_actions) != (POLICY_DIM, FRAME_DIM, ACTION_DIM):
            raise ValueError("Unsupported Lope observation/action shape")
        self.encoder = copy.deepcopy(policy.student_moe_encoder).cpu().eval()
        self.actor = copy.deepcopy(policy.actor).cpu().eval()
        self.history_normalizer = copy.deepcopy(policy.actor_obs_normalizer).cpu().eval()
        self.frame_normalizer = copy.deepcopy(policy.single_obs_normalizer).cpu().eval()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 2 or history.shape[-1] != POLICY_DIM:
            raise ValueError("Expected batch x 450 frame-major Lope history")
        latent, _ = self.encoder(self.history_normalizer(history))
        return self.actor(torch.cat((latent, self.frame_normalizer(history[:, -FRAME_DIM:])), dim=-1))


def export_jit(policy, path, filename="policy.pt"):
    destination = Path(path) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    module = LopeHistoryPolicy(policy).eval()
    torch.jit.script(module).save(str(destination))
    return destination


def export_onnx(policy, path, filename="policy.onnx", verbose=False):
    from rsl_rl.moe_cts.exporter import OnnxMoeCtsExporter

    module = LopeHistoryPolicy(policy).eval()
    module.num_actor_obs = POLICY_DIM
    module.verbose = verbose
    OnnxMoeCtsExporter.export(module, str(path), filename)
    return Path(path) / filename
