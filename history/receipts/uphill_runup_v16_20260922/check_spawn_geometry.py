"""CPU geometry probe with the saved Thunder URDF and current reset-angle range."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from reset_geometry import CollisionGeometry  # noqa: E402
from slope_runup import place  # noqa: E402

torch.set_num_threads(2)
manifest = json.loads((HERE.parent / "early_ablation_v15_20260921/evaluation_audit_20260922/terrain83/manifest.json").read_text())
urdf = HERE.parent / "wave_reset_20260919/thunder_v4.urdf"
if hashlib.sha256(urdf.read_bytes()).hexdigest() != manifest["urdf_sha256"]:
    raise RuntimeError("Local geometry is not the evaluated robot")
geometry = CollisionGeometry(urdf, manifest["joint_names"])
n = 128
generator = torch.Generator().manual_seed(922)
q = torch.tensor(manifest["default_q"])[None] * (torch.rand(n, 16, generator=generator) + .5)
q[:2] = torch.tensor(manifest["default_q"])[None] * torch.tensor([.5, 1.5])[:, None]
class Robot:
    data = NS(default_root_state=torch.zeros(n, 13), joint_pos=q)
    def write_root_pose_to_sim(self, pose, env_ids):
        self.pose = pose
    def write_root_velocity_to_sim(self, velocity, env_ids):
        pass
class Scene(dict):
    env_origins = torch.zeros(n, 3)
env = NS(scene=Scene(robot=Robot()), device="cpu", command_manager=NS(get_term=lambda _: NS(pit=NS(geometry=geometry))))
receipt = place(env, torch.arange(n), torch.tensor([.1, 1.5] * (n // 2)), torch.arange(n) % 4)
receipt.update(urdf_sha256=manifest["urdf_sha256"], samples=n, joint_reset_scale=[.5, 1.5])
(HERE / "geometry_probe.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
