"""Check the launched guarded continuation without changing its state."""
import json
import math
from pathlib import Path


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/actuator_guard_20260918'
launch = json.loads((audit / 'formal_launch.json').read_text())
output = Path(launch['output'])
assert str(output) == str(root / 'runs/official4096_30000_rs0402guard_20260918')
assert Path(f"/proc/{launch['pid']}").exists()
run = json.loads((output / 'run.json').read_text())
assert run['actuator_profile'] == 'rs0402_guard_v1'
assert run['target_completed_updates'] == 30000
assert run['starting_completed_updates'] == launch['starting_completed_updates']
assert run['resume_sha256'] == launch['checkpoint_sha256']
assert all(run['resume_checks'].values())
assert run['terrain_training_profile'] == 'stairs80_v1'
assert run['base_height_penalty'] == 'off'
assert run['stair_command_profile'] == 'straight_speed_v1'
contract = json.loads((output / 'contract.json').read_text())
assert contract['policy'] == [4096, 530]
rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
assert len(rows) >= 10, f'Only {len(rows)} updates yet'
for row in rows:
    assert all(math.isfinite(row[key]) for key in ('value', 'surrogate', 'entropy', 'mean_latent_loss'))
    assert row['actuator_guard']['samples_per_joint'] == 4096 * 24 * 4
    for name, joint in row['actuator_guard']['joints'].items():
        limit = 6.0 if name.endswith('foot_joint') else 35.0
        assert joint['peak_torque'] <= limit + 0.001
        assert joint['torque_violation_samples'] == 0
        assert joint['max_target_position_excess'] < 1e-6
latest = rows[-1]
result = {'implementation_verified': True, 'not_motion_acceptance': True,
          'pid': launch['pid'], 'starting_completed_updates': launch['starting_completed_updates'],
          'completed_updates': latest['completed_updates'], 'target_completed_updates': 30000,
          'mean_last10_seconds': sum(row['seconds'] for row in rows[-10:])/10,
          'resume_checks': run['resume_checks'], 'terrain': latest['terrain'],
          'terrain_by_type': latest['terrain_by_type'], 'actuator_guard': latest['actuator_guard'],
          'peak_speed_any_logged_update': {kind: max(joint['peak_speed'] for row in rows
              for name, joint in row['actuator_guard']['joints'].items() if name.endswith(kind+'_joint'))
              for kind in ('hip', 'thigh', 'calf', 'foot')}}
(audit / 'formal_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key not in ('terrain_by_type', 'actuator_guard')}, indent=2))
