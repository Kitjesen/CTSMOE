"""Deployment graph for a 450-value Lope history and 12 position actions."""
import copy
import os
from pathlib import Path

import torch

from lope_rl_lab.assets.lope_cfg import ACTION_DIM
from lope_rl_lab.tasks.locomotion.config import FRAME_DIM, POLICY_DIM


class LopeHistoryPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        if (policy.num_actor_obs, policy.num_single_obs, policy.num_actions) != (POLICY_DIM, FRAME_DIM, ACTION_DIM):
            raise ValueError("Unsupported Lope observation/action shape")
        self.num_actor_obs = int(POLICY_DIM)
        self.num_single_obs = int(FRAME_DIM)
        self.encoder = copy.deepcopy(policy.student_moe_encoder).cpu().eval()
        self.actor = copy.deepcopy(policy.actor).cpu().eval()
        self.history_normalizer = copy.deepcopy(policy.actor_obs_normalizer).cpu().eval()
        self.frame_normalizer = copy.deepcopy(policy.single_obs_normalizer).cpu().eval()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 2 or history.shape[-1] != self.num_actor_obs:
            raise ValueError("Expected batch x 450 frame-major Lope history")
        latent, _ = self.encoder(self.history_normalizer(history))
        current = history[:, history.shape[-1] - self.num_single_obs :]
        return self.actor(torch.cat((latent, self.frame_normalizer(current)), dim=-1))


def export_jit(policy, path, filename="policy.pt"):
    destination = Path(path) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    module = LopeHistoryPolicy(policy).eval()
    torch.jit.script(module).save(str(destination))
    return destination


def export_onnx(policy, path, filename="policy.onnx", verbose=False):
    import onnx

    destination = Path(path) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_path = str(destination)
    module = LopeHistoryPolicy(policy).eval().to("cpu")
    onnx_program = torch.onnx.export(
        module,
        torch.zeros(1, module.num_actor_obs),
        f=None,
        dynamo=True,
        export_params=True,
        external_data=False,
        opset_version=18,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions"],
    )
    if onnx_program is None:
        raise RuntimeError("The ONNX exporter did not return an ONNXProgram.")
    onnx_program.model.ir_version = 9
    onnx.checker.check_model(onnx_program.model_proto)
    onnx_program.save(export_path, external_data=False)
    external_data_path = f"{export_path}.data"
    if os.path.isfile(external_data_path):
        os.remove(external_data_path)
    return destination
