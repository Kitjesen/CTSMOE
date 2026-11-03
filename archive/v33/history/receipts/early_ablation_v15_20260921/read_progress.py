"""Compact read-only pipeline progress."""
import json
from pathlib import Path
N=Path('/root/cts-early-restart-v14-20260921')
status=json.loads((N/'pipeline_status.json').read_text())
result={'pipeline':status}
for name in ('smoke16','pilot300','continuation1000'):
    path=N/'runs'/name
    if not (path/'metrics.jsonl').exists(): continue
    rows=[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines() if x.strip()]
    if not rows:continue
    last=rows[-1]
    result[name]=dict(rows=len(rows),completed_updates=last['completed_updates'],recent_seconds=sum(x['seconds'] for x in rows[-10:])/len(rows[-10:]),collect_seconds=last['collect_seconds'],learn_seconds=last['learn_seconds'],early_restart=last['early_restart'],terrain_by_type=last['terrain_by_type'])
    validation=path/'pipeline_validation.json'
    if validation.exists():result[name]['validation']=json.loads(validation.read_text())
print(json.dumps(result,indent=2))

