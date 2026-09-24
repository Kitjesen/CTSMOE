import json
import math
import os
from pathlib import Path

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
run = root/'runs/mixed9_4096_60000_uphill_speed_20260920'
receipt = json.loads((run/'run.json').read_text())
handoff = json.loads((root/'uphill_speed_switch.json').read_text())
rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
row = rows[-1]
os.kill(handoff['new_pid'], 0)
assert all(receipt['resume_checks'].values())
assert receipt['starting_completed_updates'] == handoff['completed_updates']
assert receipt['target_completed_updates'] == 60000
assert row['completed_updates'] > handoff['completed_updates']
assert receipt['uphill_speed_limit'] == 1.2
assert row['uphill_speed']['sampled_min'][0] < -1.
assert row['uphill_speed']['sampled_max'][0] > 1.
assert row['uphill_speed']['effective_x_ranges'][0][1] > 1.19
assert row['flat_speed']['scheduled_limit_mps'] > 1.5
assert all(not any(j['torque_violation_samples'] for j in r['actuator_guard']['joints'].values()) for r in rows)
def check_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, list):
        for item in value:
            check_finite(item)
check_finite(row)
result = dict(completed_updates=row['completed_updates'], seconds=row['seconds'],
              starting_completed_updates=handoff['completed_updates'],
              resume_checks=receipt['resume_checks'], uphill_speed=row['uphill_speed'],
              flat_speed=row['flat_speed'], original_command_ranges=row['command_ranges'],
              counts=row['terrain_allocation']['actual_counts'])
(run/'uphill_speed_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
