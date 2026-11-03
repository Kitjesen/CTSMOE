"""Read-only checkpoint audit: actual parameter updates and optimizer state."""
import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
source = torch.load(a.source, map_location="cpu", weights_only=False)
candidate = torch.load(a.candidate, map_location="cpu", weights_only=False)
result = {"completed_updates": candidate["completed_updates"], "families": {}}
for prefix in ("actor.", "critic.", "teacher_encoder.", "student_moe_encoder."):
    deltas = []
    for name, value in candidate["model_state_dict"].items():
        if name.startswith(prefix) and value.is_floating_point():
            assert torch.isfinite(value).all(), name
            deltas.append(float((value-source["model_state_dict"][name]).abs().max()))
    assert deltas and max(deltas) > 0, prefix
    result["families"][prefix] = {"max_abs_delta": max(deltas), "changed_tensors": sum(d > 0 for d in deltas)}
for name in ("optimizer_state_dict", "optimizer_stu_enc_state_dict"):
    states = candidate[name]["state"]
    assert states, name
    result[name] = {"state_count": len(states), "max_step": max(float(s["step"]) for s in states.values())}
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
