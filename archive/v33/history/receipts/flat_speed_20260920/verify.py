import json
import os
from pathlib import Path

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
smoke = json.loads((root/'runs/flat_speed_smoke_20260920/run.json').read_text())
inherit = json.loads((root/'runs/flat_speed_inherit_20260920/run.json').read_text())
assert smoke['flat_speed_course'] == inherit['flat_speed_course']
assert inherit['flat_speed']['scheduled_limit_mps'] > smoke['flat_speed']['scheduled_limit_mps']
run = root/'runs/mixed9_4096_60000_flat_speed_20260920'
receipt = json.loads((run/'run.json').read_text())
rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
row = rows[-1]
os.kill(int((root/'flat_speed_formal.pid').read_text()), 0)
assert all(receipt['resume_checks'].values())
assert receipt['starting_completed_updates'] == 37283
assert receipt['target_completed_updates'] == 60000
assert row['completed_updates'] > 37283
assert row['flat_speed']['count'] == 123
assert row['flat_speed']['sampled_min'][0] < -1.
assert row['flat_speed']['sampled_max'][0] > 1.
assert all(not any(j['torque_violation_samples'] for j in r['actuator_guard']['joints'].values()) for r in rows)
result = dict(completed_updates=row['completed_updates'], seconds=row['seconds'],
              resume_checks=receipt['resume_checks'], flat_speed=row['flat_speed'],
              original_command_ranges=row['command_ranges'],
              stair_vx=row['stair_speed']['vx_min_max'],
              counts=row['terrain_allocation']['actual_counts'],
              ramp_inheritance_verified=True)
(run/'flat_speed_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
