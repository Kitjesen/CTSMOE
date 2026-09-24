import torch
from pathlib import Path

p = Path(
    "/root/autodl-tmp/cts-omni-repair-v19-20260922/yaw_stairs_v32_20260924/training_39162_to_46194/model_43500.pt"
)
s = torch.load(p, map_location="cpu", weights_only=False)


def scan(v, path=""):
    if isinstance(v, torch.Tensor):
        if path.count(".") < 2 or (v.ndim and v.shape[0] == 4096):
            print(path, tuple(v.shape))
    elif isinstance(v, dict):
        for k, x in v.items():
            scan(x, path + "." + str(k))


scan(s)
print("keys", list(s))
print("pending", s["terrain_allocation"]["pending"])
