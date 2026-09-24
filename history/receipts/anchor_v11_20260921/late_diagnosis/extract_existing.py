"""Extract named first-episode cases without changing the archived evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
m = json.loads((a.source / "manifest.json").read_text())
selected = [c for c in m["cases"] if (
    (c["terrain"] == "flat" and c["mode"] == "direct"
     and c["direction"] in ("forward", "diagonal_back"))
    or (c["terrain"] == "stairs_up" and c["level"] == 9 and c["velocity"][0] == .3)
    or (c["terrain"] == "obstacles" and c["level"] == 6 and c["velocity"][0] == 1.2)
)]
assert len(selected) == 16
ids = [c["id"] for c in selected]
data = {}
with np.load(a.source / "traces.npz") as f:
    for key in f.files:
        v = f[key]
        if key in ("initial_q", "initial_root"):
            data[key] = v[ids]
        else:
            assert v.shape[1] == len(m["cases"]), (key, v.shape)
            data[key] = v[:, ids]
data["ids"] = np.asarray(ids)
a.output.mkdir()
np.savez_compressed(a.output / "traces.npz", **data)
m["cases"] = selected
m["source_evaluation"] = str(a.source)
m["axis_case_ids"] = ids
(a.output / "manifest.json").write_text(json.dumps(m, indent=2))
print(json.dumps({"count": len(ids), "ids": ids, "output": str(a.output)}))
