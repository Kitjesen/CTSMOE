import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root=Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
names=("mixed9_4096_60000_20260919","mixed9_4096_60000_relax45_20260919")
result={}
for name in names:
 p=root/"runs"/name
 rows=[json.loads(x) for x in (p/"metrics.jsonl").read_text().splitlines()]
 a,b=rows[-101],rows[-1]
 out={"start":a["completed_updates"],"end":b["completed_updates"],"terrain":b["terrain"],
      "mean_levels_last100":{k:sum(x["terrain_by_type"][k]["mean"] for x in rows[-100:])/100 for k in b["terrain_by_type"]}}
 out["stairs"]={}
 for learner in ("teacher","student"):
  out["stairs"][learner]={}
  for direction in ("stairs_up","stairs_down"):
   aa=a["stair_speed"]["episode_statistics"][learner][direction];bb=b["stair_speed"]["episode_statistics"][learner][direction]
   counts={k:sum(y[k]-x[k] for x,y in zip(aa,bb)) for k in ("episodes","grade_eligible","radius_passes","physical_failures")}
   counts["vx_mae_mps"]=sum(y["vx_mae_mps"]*y["episodes"]-x["vx_mae_mps"]*x["episodes"] for x,y in zip(aa,bb))/counts["episodes"]
   out["stairs"][learner][direction]=counts
 out["pit"]={}
 for learner in ("teacher","student"):
  aa=a["pit"]["by_learner"][learner]["totals"];bb=b["pit"]["by_learner"][learner]["totals"]
  out["pit"][learner]={k:bb[k]-aa[k] for k in bb}
 out["pit_frontier"]=b["pit"]["frontier_height_m"]
 out["guard"]={k:sum(j[k] for row in rows[-100:] for j in row["actuator_guard"]["joints"].values()) for k in ("overspeed_samples","torque_violation_samples")}
 acc=EventAccumulator(str(p),size_guidance={"scalars":0});acc.Reload()
 out["scalar_last100"]={}
 for tag in acc.Tags()["scalars"]:
  if any(term in tag.lower() for term in ("termination","episode_length","mean_reward","learning_rate","entropy","kl")):
   vals=acc.Scalars(tag)[-100:]
   out["scalar_last100"][tag]=sum(x.value for x in vals)/len(vals)
 result[name]=out
(root/"relax45_progress_1455.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

