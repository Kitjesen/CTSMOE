"""Require actual simulation, finite updates, terminal episodes, and exact restore."""
import json
import math
from pathlib import Path

import torch


audit = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/stair_speed_20260918')
first = json.loads((audit / 'smoke64/run.json').read_text())
resume = json.loads((audit / 'resume64/run.json').read_text())
assert first['stair_command_profile'] == resume['stair_command_profile'] == 'straight_speed_v1'
assert first['resume_checks']['terrain_restored_exactly']
assert resume['resume_checks']['terrain_restored_exactly']
assert resume['resume_checks']['stair_episode_totals']
assert not resume['initial_episode_lengths_randomized']
for name in ('model', 'ppo_optimizer', 'student_optimizer', 'terrain_assignment'):
    assert first['resume_checks'][name] and resume['resume_checks'][name]
assert json.loads((audit / 'smoke64/SUCCESS.json').read_text())['completed_updates'] == 68
assert json.loads((audit / 'resume64/SUCCESS.json').read_text())['completed_updates'] == 72
rows = [json.loads(line) for line in (audit / 'smoke64/metrics.jsonl').read_text().splitlines()]
assert len(rows) == 60
for row in rows:
    assert row['stair_speed']['vy_wz_abs_max'] == [0., 0.]
    assert min(row['stair_speed']['active_speed_bin_counts']) > 0
    assert row['base_height_penalty_weight'] == 0
    for key in ('value', 'surrogate', 'entropy', 'mean_latent_loss', 'mean_load_balance_loss'):
        assert math.isfinite(row[key])
saved = torch.load(audit / 'smoke64/model_67.pt', map_location='cpu', weights_only=False)
assert saved['stair_episode_totals'][..., 0].sum() > 0
assert saved['stair_command_profile'] == 'straight_speed_v1'
result = {'checks_passed': True, 'updates': len(rows),
          'completed_stair_episodes': int(saved['stair_episode_totals'][..., 0].sum()),
          'last': rows[-1], 'resume_checks': resume['resume_checks']}
(audit / 'smoke_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
