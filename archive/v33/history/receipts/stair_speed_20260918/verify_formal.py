"""Read-only checks of the newly running straight-speed continuation."""
import datetime
import json
import math
from pathlib import Path
import statistics

import torch


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/stair_speed_20260918'
run = root / 'runs/official4096_30000_straightspeed_20260918'
receipt = json.loads((run / 'run.json').read_text())
source = json.loads((audit / 'switch_source.json').read_text())
assert receipt['resume_sha256'] == source['checkpoint_sha256']
assert receipt['target_completed_updates'] == 30000
assert receipt['stair_command_profile'] == 'straight_speed_v1'
assert receipt['terrain_training_profile'] == 'stairs80_v1'
assert receipt['base_height_penalty'] == 'off'
assert receipt['reward_weights'] == {'lin_vel_z_l2': 0., 'base_height_l2': 0.}
assert not receipt['initial_episode_lengths_randomized']
for name in ('model', 'ppo_optimizer', 'student_optimizer', 'terrain_assignment',
             'command_terrain_membership', 'terrain_restored_exactly'):
    assert receipt['resume_checks'][name], name
saved = torch.load(source['checkpoint'], map_location='cpu', weights_only=False)
assert sum(x['count'] for x in receipt['terrain_by_type'].values()) == 4096
assert receipt['terrain_by_type']['stairs_up']['count'] == 2457
assert receipt['terrain_by_type']['stairs_down']['count'] == 819
for name, stats in receipt['terrain_by_type'].items():
    levels = [int(level) for level, column in zip(saved['terrain_levels'], saved['terrain_types'])
              if receipt['terrain_column_names'][int(column)] == name]
    assert stats['counts_L0_to_L9'] == [levels.count(level) for level in range(10)]
rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
assert len(rows) >= 12, len(rows)
for row in rows:
    speed = row['stair_speed']
    assert speed['stairs_command_count'] == 3276
    assert speed['vy_wz_abs_max'] == [0., 0.]
    assert speed['vx_min_max'][0] >= 0.2 and speed['vx_min_max'][1] <= 1.0
    assert min(speed['active_speed_bin_counts']) > 0
    assert row['base_height_penalty_weight'] == 0
    for name in ('value', 'surrogate', 'entropy', 'mean_latent_loss', 'mean_load_balance_loss'):
        assert math.isfinite(row[name])
result = {'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'checks_passed': True, 'new_updates': len(rows),
          'starting_completed_updates': receipt['starting_completed_updates'],
          'starting_terrain_mean': source['terrain_mean_before_resume'],
          'seconds_mean_last10': statistics.mean(row['seconds'] for row in rows[-10:]),
          'last': rows[-1], 'resume_checks': receipt['resume_checks']}
(audit / 'formal_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps({key: result[key] for key in ('verified_at_utc', 'checks_passed', 'new_updates',
                                              'starting_completed_updates', 'starting_terrain_mean',
                                              'seconds_mean_last10', 'resume_checks')}))
