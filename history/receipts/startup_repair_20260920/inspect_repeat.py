import json
from pathlib import Path

r=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
p=r/'runs/startup_repair_repeat96_20260920'
state=json.loads((p/'status.json').read_text())
rows=(p/'metrics.jsonl').read_text().splitlines()
last=json.loads(rows[-1])
print(json.dumps(dict(state=state['state'],completed=last['completed_updates'],
    final=43106,seconds=sum(json.loads(x)['seconds'] for x in rows[-10:])/len(rows[-10:]),
    repair=last['startup_repair'],
    audit=(r/'recording/startup_repair_repeat96_20260920/manifest.json').is_file(),
    regression=(r/'recording/startup_repair_repeat_regression96_20260920/manifest.json').is_file()),indent=2))
