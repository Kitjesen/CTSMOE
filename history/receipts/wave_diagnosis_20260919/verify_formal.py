"""Verify the live 4096-env v3 continuation and preserved restoration state."""

import json
import math
from pathlib import Path


root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root / 'recording/wave_diagnosis_20260919'
launch = json.loads((folder / 'formal_launch.json').read_text())
run_folder = Path(launch['output'])
assert Path(f"/proc/{launch['pid']}").exists()
run = json.loads((run_folder / 'run.json').read_text())
assert run['actuator_profile'] == 'rs0402_slew12_v3'
assert run['leg_target_max_delta_rad'] == 0.24
assert run['target_completed_updates'] == 30000
assert run['starting_completed_updates'] == launch['starting_completed_updates']
assert run['resume_sha256'] == launch['checkpoint_sha256']
assert all(run['resume_checks'].values())
assert run['terrain_training_profile'] == 'stairs80_v1'
assert run['base_height_penalty'] == 'off'
assert run['stair_command_profile'] == 'straight_speed_v1'
assert json.loads((run_folder / 'contract.json').read_text())['policy'] == [4096, 530]
rows = [json.loads(line) for line in (run_folder / 'metrics.jsonl').read_text().splitlines()]
assert len(rows) >= 10, len(rows)
for row in rows:
    assert row['actuator_profile'] == 'rs0402_slew12_v3'
    assert row['actuator_guard']['limits']['leg_target_rate_rad_s'] == 12.0
    assert all(math.isfinite(row[key]) for key in ('value', 'surrogate', 'entropy', 'mean_latent_loss'))
    assert row['actuator_guard']['samples_per_joint'] == 4096 * 24 * 4
    for name, joint in row['actuator_guard']['joints'].items():
        assert joint['peak_torque'] <= (17.0 if name.endswith('foot_joint') else 120.0) + 0.001
        assert joint['torque_violation_samples'] == 0
        assert joint['max_target_position_excess'] < 1e-6
result = dict(verified=True, pid=launch['pid'], starting_completed_updates=run['starting_completed_updates'],
              completed_updates=rows[-1]['completed_updates'], updates_observed=len(rows),
              target_completed_updates=30000, leg_target_max_delta_rad=0.24,
              resume_checks=run['resume_checks'], terrain=rows[-1]['terrain'],
              terrain_by_type=rows[-1]['terrain_by_type'],
              seconds_mean_last10=sum(row['seconds'] for row in rows[-10:])/10,
              limitations='Training launch and accounting verified; frozen v2 policy did not improve simply by increasing slew. No automatic reduction schedule is enabled.')
(folder / 'formal_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
