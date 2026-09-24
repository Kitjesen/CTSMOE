"""Read-only deployment evidence, including stage activity and curriculum migration."""
import hashlib
import json
from pathlib import Path

root=Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
smoke=root/'runs/phase_v8_smoke_20260921'
run=json.loads((smoke/'run.json').read_text())
assert json.loads((smoke/'SUCCESS.json').read_text())['completed_updates']==43218
metrics=[json.loads(line) for line in (smoke/'metrics.jsonl').read_text().splitlines()]
assert len(metrics)==16
assert any(row['phase_transition']['recovering']>0 for row in metrics)
assert any(row['phase_transition']['active_climb']>0 for row in metrics)
assert run['resume_checks']['model'] and run['resume_checks']['ppo_optimizer'] and run['resume_checks']['student_optimizer']
assert run['startup_repair_profile']=='omni_start_v8'
assert run['resume_checks']['explicit_pit_foundation_migration']
receipt=dict(smoke_updates=16,smoke_completed=43218,formal_target=44202,
             resume_checks=run['resume_checks'],
             foundation=json.loads((smoke/'phase_foundation_migration.json').read_text()),
             last_smoke_phase=metrics[-1]['phase_transition'],
             last_smoke_seconds=metrics[-1]['seconds'],
             code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (root/'code_phase_v8_20260921').glob('*.py')})
(root/'phase_v8_deployment_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='code_hashes'},indent=2))
