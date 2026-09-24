"""Audit completed preflights against the saved source and matched wave cases."""
import json
from pathlib import Path
import sys
import torch

root = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
sys.path.insert(0, str(root/"code_relax45_20260919_v1"))
from training_profiles import summarize_terrain
source = torch.load(root/"runs/mixed9_4096_60000_20260919/model_30345.pt", map_location="cpu", weights_only=False)
trained = torch.load(root/"runs/relax45_smoke4096_20260919/model_30349.pt", map_location="cpu", weights_only=False)
run = json.loads((root/"runs/relax45_smoke4096_20260919/run.json").read_text())
before = summarize_terrain(source["terrain_levels"].tolist(), source["terrain_types"].tolist(), source["terrain_column_names"])
unchanged = []
for name in ("wave","stairs_up","stairs_down","obstacles","flat","pit"):
    assert before[name] == run["terrain_by_type"][name], name
    unchanged.append(name)
updates = {}
for prefix in ("actor.", "teacher_encoder.", "student_moe_encoder."):
    keys = [key for key in source["model_state_dict"] if key.startswith(prefix)]
    assert keys, prefix
    differences = [(trained["model_state_dict"][key]-source["model_state_dict"][key]).abs().max().item() for key in keys]
    assert max(differences) > 0, prefix
    updates[prefix] = max(differences)
for value in trained["model_state_dict"].values():
    assert torch.isfinite(value).all()
stats = {}
for name in ("relax45_smoke4096_20260919","relax45_inherit4096_20260919"):
    path = root/"runs"/name
    assert json.loads((path/"SUCCESS.json").read_text())["state"] == "complete"
    receipt = json.loads((path/"run.json").read_text())
    assert receipt["actuator_profile"] == "rs0402_transient_v4" and receipt["slope_profile"] == "slope45_v1"
    if "inherit" in name:
        assert all(receipt["resume_checks"].values())
    rows = [json.loads(line) for line in (path/"metrics.jsonl").read_text().splitlines()]
    joints = [j for row in rows for j in row["actuator_guard"]["joints"].values()]
    assert sum(j["torque_violation_samples"] for j in joints) == 0
    stats[name] = dict(completed=rows[-1]["completed_updates"],
        seconds_mean=sum(row["seconds"] for row in rows)/len(rows),
        tolerated_overspeed_samples=sum(j["tolerated_overspeed_samples"] for j in joints),
        sustained_overspeed_samples=sum(j["sustained_overspeed_samples"] for j in joints),
        torque_violation_samples=0)
waves = [json.loads((root/"runs"/("relax45_wave_"+v+"_20260919")/"results.json").read_text()) for v in ("v3","v4")]
assert waves[0]["initial_root"] == waves[1]["initial_root"]
assert waves[0]["initial_joints"] == waves[1]["initial_joints"]
ab = {}
for learner in ("teacher","student"):
    ab[learner] = {}
    for version, wave in zip(("v3","v4"), waves, strict=True):
        cases = [c for c in wave["cases"] if c["learner"] == learner]
        ab[learner][version] = dict(n=len(cases), success=sum(c["success"] for c in cases),
            physical=sum(c["physical"] for c in cases),
            overspeed=sum(c.get("causes",{}).get("speed",False) for c in cases),
            early_physical=sum(c["physical"] and c["seconds"]<=1. for c in cases),
            by_level={level:dict(n=sum(c["level"]==level for c in cases),
                success=sum(c["success"] and c["level"]==level for c in cases)) for level in (0,3,6,9)})
result = dict(source_completed=source["completed_updates"], unchanged_non_slope_types=unchanged,
              max_parameter_changes=updates, training=stats, matched_initial_states=True, wave_ab=ab)
(root/"relax45_validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

