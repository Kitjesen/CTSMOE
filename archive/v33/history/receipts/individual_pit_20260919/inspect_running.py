import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
p = root/'runs/mixed9_4096_60000_individualpit_20260919'
pid = int((root/'individual_formal.pid').read_text())
os.kill(pid, 0)
run = json.loads((p/'run.json').read_text())
rows = [json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
row = json.loads((p/'status.json').read_text())
assert row['state'] == 'training'
assert all(run['resume_checks'].values())
assert run['target_completed_updates'] == 60000
assert row['pit_course_profile'] == 'individual_speed_v1'
assert sum(row['terrain_allocation']['actual_counts'].values()) == 4096
assert all(sum(j['torque_violation_samples'] for j in r['actuator_guard']['joints'].values()) == 0 for r in rows)
result = dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), pid=pid,
              state=row['state'], completed_updates=row['completed_updates'],
              start=run['starting_completed_updates'], target=run['target_completed_updates'],
              mean_seconds=sum(r['seconds'] for r in rows[-10:])/len(rows[-10:]),
              resume_checks=run['resume_checks'], individual=row['pit']['course'],
              counts=row['terrain_allocation']['actual_counts'])
(p/'runtime_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
