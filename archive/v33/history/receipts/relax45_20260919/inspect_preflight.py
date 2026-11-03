"""Read-only status of the relaxation and slope migration preflight."""
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone, timedelta
root = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
print(datetime.now(timezone(timedelta(hours=8))).isoformat())
for name in ("relax45_smoke4096_20260919", "relax45_inherit4096_20260919",
             "relax45_wave_v3_20260919", "relax45_wave_v4_20260919",
             "mixed9_4096_60000_relax45_20260919"):
    path = root/"runs"/name
    if not path.exists():
        print(name, "not_started")
        continue
    print("RUN", name)
    if (path/"status.json").exists():
        state = json.loads((path/"status.json").read_text())
        print(json.dumps({k:state[k] for k in ("state","completed_updates","seconds","actuator_profile","slope_profile","terrain","error") if k in state}))
    if (path/"run.json").exists():
        run = json.loads((path/"run.json").read_text())
        print("RESUME", run["resume_checks"])
        print("PROFILE", run["actuator_profile"], run["slope_profile"])
        if "smoke" in name:
            print("GEOMETRY",json.dumps(run["mixed_checks"]["slope_geometry"]))
            print("MIGRATION", (path/"slope_migration.json").read_text())
    if (path/"results.json").exists():
        result = json.loads((path/"results.json").read_text())
        for learner in ("teacher","student"):
            cases = [c for c in result["cases"] if c["learner"] == learner]
            print("WAVE", learner, "success",sum(c["success"] for c in cases),
                  "physical",sum(c["physical"] for c in cases), "n",len(cases))
    print("CHECKPOINTS", sorted(p.name for p in path.glob("model_*.pt")))
print(subprocess.run(["pgrep","-af","train.py|probe_wave_relax45.py|relax45_preflight.sh"],capture_output=True,text=True).stdout)
print(subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.free","--format=csv,noheader"],capture_output=True,text=True).stdout)
