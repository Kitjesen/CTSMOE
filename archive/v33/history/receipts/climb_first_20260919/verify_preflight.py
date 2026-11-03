import json
from pathlib import Path
import torch
root=Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
result={}
for name in ("climbfirst_smoke4096_20260919","climbfirst_inherit4096_20260919"):
 p=root/"runs"/name
 assert json.loads((p/"SUCCESS.json").read_text())["state"]=="complete"
 run=json.loads((p/"run.json").read_text())
 assert all(run["resume_checks"].values())
 assert run["pit_training_profile"]=="climb_first_v2"
 assert run["actuator_profile"]=="rs0402_transient_v4" and run["slope_profile"]=="slope45_v1"
 rows=[json.loads(x) for x in (p/"metrics.jsonl").read_text().splitlines()]
 assert all(len(x["terrain_by_type"])==9 for x in rows)
 assert all(sum(j["torque_violation_samples"] for j in x["actuator_guard"]["joints"].values())==0 for x in rows)
 result[name]={"completed_updates":rows[-1]["completed_updates"],"resume_checks":run["resume_checks"],
               "last_pit":rows[-1]["pit"]["climb_by_learner"],"course":rows[-1]["pit"]["course"],
               "mean_seconds":sum(x["seconds"] for x in rows)/len(rows),
               "old_terrain_reward_isolation":run["mixed_checks"]["pit_rewards_zero_on_old"]}
for index in (31862,31864):
 name="climbfirst_smoke4096_20260919" if index==31862 else "climbfirst_inherit4096_20260919"
 saved=torch.load(root/"runs"/name/("model_"+str(index)+".pt"),weights_only=False,map_location="cpu")
 assert saved["pit_training_profile"]=="climb_first_v2"
 assert all(torch.isfinite(x).all() for x in saved["model_state_dict"].values())
result["checkpoint_contracts_and_finite_parameters"]=True
(root/"climbfirst_validation.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

