"""Compact status for a bounded repair experiment, not the stopped main run."""
import json
from pathlib import Path

r=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
p=r/'runs/startup_repair_trial96_20260920'
state=json.loads((p/'status.json').read_text())
rows=(p/'metrics.jsonl').read_text().splitlines()
last=json.loads(rows[-1])
receipt=json.loads((p/'run.json').read_text())
result=dict(state=state['state'],completed=last['completed_updates'],
            planned_final=receipt['target_completed_updates'],
            mean_seconds=sum(json.loads(row)['seconds'] for row in rows[-10:])/len(rows[-10:]),
            resume_checks=receipt['resume_checks'],repair=last['startup_repair'],
            allocation=last['terrain_allocation'],
            losses={k:last[k] for k in ('value','surrogate','mean_latent_loss')},
            candidate_audit=(r/'recording/startup_repair_candidate96_20260920/manifest.json').is_file(),
            regression=(r/'recording/startup_repair_regression96_20260920/manifest.json').is_file())
(r/'startup_repair_trial_status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
