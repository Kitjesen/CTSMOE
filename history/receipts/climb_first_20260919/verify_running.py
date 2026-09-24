import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
p=Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/runs/mixed9_4096_60000_climbfirst_20260919")
run=json.loads((p/"run.json").read_text())
rows=[json.loads(x) for x in (p/"metrics.jsonl").read_text().splitlines()]
s=rows[-1]
os.kill(136431,0)
assert s["state"]=="training" and s["completed_updates"]>31865
assert s["pit_training_profile"]=="climb_first_v2"
assert all(run["resume_checks"].values())
assert len(s["terrain_by_type"])==9
assert all(sum(j["torque_violation_samples"] for j in row["actuator_guard"]["joints"].values())==0 for row in rows)
result=dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),pid=136431,state=s["state"],
    completed_updates=s["completed_updates"],target=run["target_completed_updates"],num_envs=sum(x["count"] for x in s["terrain_by_type"].values()),
    seconds_mean=sum(x["seconds"] for x in rows[-10:])/len(rows[-10:]),
    frontier_m=s["pit"]["frontier_height_m"],climbs=s["pit"]["climb_by_learner"],
    resume_checks=run["resume_checks"],torque_violations=0)
(p/"runtime_verified.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

