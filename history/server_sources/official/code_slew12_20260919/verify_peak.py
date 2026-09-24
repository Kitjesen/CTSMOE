"""Verify peak-budget migration, inherited resume, export and playback."""
import json
import math
from pathlib import Path

import torch

from policy_contract import checkpoint_profile, export_contract, load_student_export


audit = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917/recording/actuator_peak_20260918')
contract = export_contract('rs0402_peak_v2')
result = {}
for folder, count, updates in (('smoke64', 88, 4), ('resume64', 90, 2)):
    output = audit / folder
    assert json.loads((output / 'SUCCESS.json').read_text())['completed_updates'] == count
    run = json.loads((output / 'run.json').read_text())
    assert run['actuator_profile'] == 'rs0402_peak_v2'
    assert all(run['resume_checks'].values())
    saved = torch.load(output / f'model_{count-1}.pt', map_location='cpu', weights_only=False)
    assert checkpoint_profile(saved) == 'rs0402_peak_v2'
    assert saved['actuator_contract'] == contract
    _, exported = load_student_export(output / f'student_{count}.pt')
    assert exported == contract
    rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert len(rows) == updates
    for row in rows:
        assert all(math.isfinite(row[key]) for key in ('value', 'surrogate', 'entropy', 'mean_latent_loss'))
        stats = row['actuator_guard']
        assert stats['limits'] == contract['limits']
        assert stats['samples_per_joint'] == 64 * 24 * 4
        for name, joint in stats['joints'].items():
            assert joint['peak_torque'] <= (17 if name.endswith('foot_joint') else 120) + 0.001
            assert joint['torque_violation_samples'] == 0
            assert joint['max_target_position_excess'] < 1e-6
    result[folder] = {'completed_updates': count, 'resume_checks': run['resume_checks']}
manifest = json.loads((audit / 'record_smoke/manifest.json').read_text())
assert manifest['actuator_contract'] == contract
result.update(passed=True, actuator_contract=contract, playback_contract_verified=True)
(audit / 'validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
