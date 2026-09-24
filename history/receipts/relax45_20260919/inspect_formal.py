"""Compact, read-only verification of the active formal run."""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import subprocess
root = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
path = root/"runs/mixed9_4096_60000_relax45_20260919"
run = json.loads((path/"run.json").read_text())
rows = [json.loads(line) for line in (path/"metrics.jsonl").read_text().splitlines()]
assert rows and rows[-1]["state"] == "training"
assert rows[-1]["completed_updates"] > run["starting_completed_updates"]
proc = subprocess.run(["ps","-p","129951","-o","args="],capture_output=True,text=True)
assert proc.returncode == 0 and str(path) in proc.stdout
for row in rows:
    assert row["actuator_profile"] == "rs0402_transient_v4" and row["slope_profile"] == "slope45_v1"
    assert sum(j["torque_violation_samples"] for j in row["actuator_guard"]["joints"].values()) == 0
assert len(rows[-1]["terrain_by_type"]) == 9
last = rows[-1]
result = dict(queried_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), pid=129951,
    state="training", completed_updates=last["completed_updates"], target_updates=run["target_completed_updates"],
    additional_verified=len(rows), seconds_last10_mean=sum(row["seconds"] for row in rows[-10:])/len(rows[-10:]),
    actuator_profile=last["actuator_profile"], slope_profile=last["slope_profile"],
    num_envs=sum(x["count"] for x in last["terrain_by_type"].values()),
    terrain_by_type=last["terrain_by_type"], resume_checks=run["resume_checks"],
    torque_violation_samples=0,
    tolerated_overspeed_samples=sum(j["tolerated_overspeed_samples"] for row in rows for j in row["actuator_guard"]["joints"].values()),
    gpu=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.free","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
(path/"runtime_verified.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

