import json
from pathlib import Path
root=Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
for name in ("climbfirst_smoke4096_20260919","climbfirst_inherit4096_20260919","mixed9_4096_60000_climbfirst_20260919"):
 p=root/"runs"/name
 if not (p/"run.json").exists():
  print(name,"initializing_or_not_started")
  continue
 run=json.loads((p/"run.json").read_text())
 print(name,"RESUME",run["resume_checks"],"PIT_PROFILE",run["pit_training_profile"])
 if (p/"status.json").exists():
  row=json.loads((p/"status.json").read_text())
  print("STATE",row["state"],row.get("completed_updates"),row.get("seconds"),row.get("error"))
  if "pit" in row:
   pit=row["pit"]
   print("PIT",json.dumps({"frontier":pit["frontier_height_m"],"climbs":pit["climb_by_learner"],"course":pit["course"]}))
 if (p/"metrics.jsonl").exists():
  rows=[json.loads(x) for x in (p/"metrics.jsonl").read_text().splitlines()]
  assert all(x["pit_training_profile"]=="climb_first_v2" for x in rows)
  assert all(sum(j["torque_violation_samples"] for j in x["actuator_guard"]["joints"].values())==0 for x in rows)
  print("VERIFIED_UPDATES",len(rows),"TORQUE_VIOLATIONS",0)

