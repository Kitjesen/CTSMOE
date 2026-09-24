"""CPU-only audit of recorded stair triggers; does not change the evaluator."""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/cts-omni-repair-v19-20260922/speed_brake_v20/requested_evaluation_33865")
manifest = json.loads((ROOT / "focused/stairend/manifest.json").read_text())
data = np.load(ROOT / "focused/stairend/traces.npz")
keys = ["active", "stair_platform_hull", "stair_platform_supported", "phase_ready",
        "ground_relief", "wheel_clearance", "wheel_ground_force_z", "command"]
arrays = {key: data[key] for key in keys}
rows = []
for case in manifest["cases"]:
    if case["policy"] != 2 or case["landing_mode"] != "stop":
        continue
    i = case["id"]
    valid = arrays["active"][:, i].astype(bool)
    if case.get("reset_metadata", {}).get("reset_occurred_after_capture", False):
        valid[case["end_step"]] = False
    hull = valid & arrays["stair_platform_hull"][:, i].astype(bool)
    contact = (arrays["wheel_ground_force_z"][:, i] > 5) & (np.abs(arrays["wheel_clearance"][:, i]) < .04)
    count = contact.sum(-1)
    relief = arrays["ground_relief"][:, i]
    rows.append({
        "id": i, "terrain": case["terrain"], "level": case["level"],
        "velocity": case["velocity"], "end_reason": case["end_reason"],
        "hull_steps": int(hull.sum()),
        "hull_scan_relief_le_5cm_steps": int((hull & (relief <= .05)).sum()),
        "hull_3near_contacts_steps": int((hull & (count >= 3)).sum()),
        "hull_scan_and_3contacts_steps": int((hull & (relief <= .05) & (count >= 3)).sum()),
        "recorded_supported_steps": int((valid & arrays["stair_platform_supported"][:, i].astype(bool)).sum()),
        "hull_scan_relief_range": [float(relief[hull].min()), float(relief[hull].max())] if hull.any() else None,
        "zero_command_steps": int((valid & (np.abs(arrays["command"][:, i]).max(-1) < 1e-7)).sum()),
        "note": "Recorded sensors may have update latency; these marginals do not prove counterfactual success without rerun.",
    })
result = {"optimizer_updates": 0, "scope": "existing immutable traces only", "cases": rows}
with (ROOT / "PLATFORM_TRIGGER_AUDIT.json").open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result))
