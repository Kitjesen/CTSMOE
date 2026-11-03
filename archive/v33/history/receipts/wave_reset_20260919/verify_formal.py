"""Verify the deployed profile, exact restoration and continued guarded updates."""
import hashlib
import json
import math
from pathlib import Path

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
folder = root/'recording/wave_reset_20260919'
launch = json.loads((folder/'formal_launch.json').read_text())
assert Path(f'/proc/{launch["pid"]}').exists()
run_folder = Path(launch['output'])
run = json.loads((run_folder/'run.json').read_text())
assert run['resume_sha256'] == launch['checkpoint_sha256']
assert all(run['resume_checks'].values())
assert run['wave_reset_profile'] == 'wave_surface_v1'
assert run['actuator_profile'] == 'rs0402_slew12_v3'
assert run['terrain_training_profile'] == 'stairs80_v1'
assert run['base_height_penalty'] == 'off'
assert run['stair_command_profile'] == 'straight_speed_v1'
assert run['leg_target_max_delta_rad'] == .24
assert run['target_completed_updates'] == 30000
for name, digest in launch['code_sha256'].items():
    assert hashlib.sha256((root/'code_wave_reset_20260919'/name).read_bytes()).hexdigest() == digest
rows = [json.loads(line) for line in (run_folder/'metrics.jsonl').read_text().splitlines()]
assert len(rows) >= 10
for row in rows:
    assert row['wave_reset']['profile'] == 'wave_surface_v1'
    assert row['wave_reset']['count'] >= run['wave_reset']['count'] > 0
    assert .0299 < row['wave_reset']['last']['collision_gap_min'] < .0301
    assert row['actuator_guard']['limits']['leg_target_rate_rad_s'] == 12
    assert row['actuator_guard']['samples_per_joint'] == 4096*24*4
    assert all(math.isfinite(row[key]) for key in ('value', 'surrogate', 'entropy', 'mean_latent_loss'))
    for name, joint in row['actuator_guard']['joints'].items():
        assert joint['peak_torque'] <= (17.0 if name.endswith('foot_joint') else 120.0)+.001
        assert joint['torque_violation_samples'] == 0
        assert joint['max_target_position_excess'] < 1e-6
result = dict(verified=True, pid=launch['pid'], completed_updates=rows[-1]['completed_updates'], updates_observed=len(rows),
              starting_completed_updates=run['starting_completed_updates'], target_completed_updates=30000,
              resume_checks=run['resume_checks'], seconds_mean_last10=sum(r['seconds'] for r in rows[-10:])/10,
              wave_reset=rows[-1]['wave_reset'], terrain=rows[-1]['terrain'], terrain_by_type=rows[-1]['terrain_by_type'],
              scope='Implementation and live continuation verified; recovery of wave terrain level requires subsequent training')
(folder/'formal_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'terrain_by_type'}, indent=2))
