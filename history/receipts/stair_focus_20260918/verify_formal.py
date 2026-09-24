"""Read-only acceptance of the active stair-focused continuation."""
import datetime
import json
import math
from pathlib import Path
import statistics

import torch
import yaml


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
run = root / 'runs/official4096_30000_stairs80_noheight_20260918'
audit = root / 'recording/stair_focus_20260918'
receipt = json.loads((run / 'run.json').read_text())
migration = json.loads((run / 'terrain_migration.json').read_text())
rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
assert len(rows) >= 10, len(rows)
assert receipt['terrain_training_profile'] == 'stairs80_v1'
assert receipt['base_height_penalty'] == 'off'
assert receipt['target_completed_updates'] == 30000
assert receipt['reward_weights']['base_height_l2'] == 0
assert receipt['reward_weights']['lin_vel_z_l2'] == 0
for name in ('model', 'ppo_optimizer', 'student_optimizer', 'terrain_assignment', 'command_terrain_membership'):
    assert receipt['resume_checks'][name]
assert not receipt['resume_checks']['terrain_restored_exactly']
counts = {name: stats['count'] for name, stats in receipt['terrain_by_type'].items()}
assert counts['stairs_up'] == 2457 and counts['stairs_down'] == 819
assert sum(counts.values()) == 4096
config = yaml.load((run / 'env.yaml').read_text(), Loader=yaml.BaseLoader)
assert float(config['rewards']['base_height_l2']['weight']) == 0
assert config['curriculum']['base_height_l2'] == 'null'
assert receipt['command_ranges'] == {'lin_vel_x': [-1., 1.], 'lin_vel_y': [-1., 1.], 'ang_vel_yaw': [-1.5, 1.5]}
for row in rows:
    assert row['base_height_penalty_weight'] == 0
    assert row['terrain_by_type']['stairs_up']['count'] == 2457
    assert row['terrain_by_type']['stairs_down']['count'] == 819
    for name in ('value', 'surrogate', 'entropy', 'mean_latent_loss', 'mean_load_balance_loss'):
        assert math.isfinite(row[name])
saved = torch.load(receipt['resume'], map_location='cpu', weights_only=False)
levels, columns = saved['terrain_levels'].tolist(), saved['terrain_types'].tolist()
column_names = receipt['terrain_column_names']
for move in migration['moves']:
    i, donor = move['env_id'], move['donor_env_id']
    assert column_names[columns[i]] == move['from']
    assert column_names[columns[donor]] == move['to']
    assert levels[donor] == move['new_level']
    assert (i % 4 == 0) == (donor % 4 == 0)
result = {'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'checks_passed': True, 'run': str(run), 'receipt': receipt,
          'new_updates_observed': len(rows), 'latest': rows[-1],
          'seconds_mean_last10': statistics.mean(row['seconds'] for row in rows[-10:]),
          'reassigned_environments': migration['reassigned_environments'],
          'terrain_before': migration['before'], 'terrain_after_assignment': migration['after']}
(audit / 'formal_verified.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key: result[key] for key in ('verified_at_utc', 'checks_passed', 'new_updates_observed',
                                               'seconds_mean_last10', 'reassigned_environments', 'latest')}, indent=2))
