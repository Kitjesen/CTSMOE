import json
from pathlib import Path
root=Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
result={}
for name in ("pit_predicates31001_20260919","pit_predicates31001_noise_20260919"):
 p=root/"runs"/name/"results.json"
 if not p.exists():
  print(name,"running")
  continue
 data=json.loads(p.read_text())
 cs=data["cases"]
 stop=sum(c["after_stop_seconds"]["stop_command"] for c in cs)
 out={"checkpoint_updates":data["completed_updates"],"n":len(cs),
 "success":sum(c["success"] for c in cs),"physical":sum(c.get("physical",False) for c in cs),
 "ever_counts":{k:sum(c["ever"][k] for c in cs) for k in cs[0]["ever"]},
 "max_hold_seconds":[c["maximum_valid_hold_seconds"] for c in cs],
 "after_stop_valid_fraction":{k:sum(c["after_stop_seconds"][k] for c in cs)/stop for k in cs[0]["ever"]},
 "by_speed":{speed:{"n":sum(c["speed"]==speed for c in cs),"success":sum(c["speed"]==speed and c["success"] for c in cs)} for speed in (.3,.5,.7,.9)},
 "final_state_ranges":{k:[min(c["last"][k] for c in cs),max(c["last"][k] for c in cs)] for k in ("rear_clearance_m","linear_speed","angular_speed","tilt_deg","command_vx")}}
 if "action_std" in data:
  out["action_std"]=data["action_std"]
 result[name]=out
 print(name,json.dumps(out,indent=2))
if len(result)==2:
 (root/"pit_predicates31001_comparison.json").write_text(json.dumps(result,indent=2))
