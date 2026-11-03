"""Qualify a serialization-only correction against the unchanged 3000-step runtime.

Do not repeat physics to test a metadata format: require identical runtime files,
prove the complete train.py delta is exactly the named serialization/restore
validation edits, independently verify the real preflight, and exercise the
actual save expression through the strict checkpoint loader. No simulation
result is copied or represented as a run of the corrected entry point.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PREFLIGHT = HERE.parent/'hip_guard_v22_attempt02'
sys.path.insert(0,str(HERE/'code'))
from hip_contract import PROFILE, contract  # noqa: E402
from policy_contract import checkpoint_profile, export_contract  # noqa: E402
from runtime_contract import verify_prepared  # noqa: E402


def prove_delta(old, corrected):
    insertion = '{**export_contract(actuator_profile), "hip_guard": hip_contract()}'
    if old.count(insertion) != 2:
        raise ValueError('Unknown serialized actuator contract change')
    expected = old.replace(insertion,'export_contract(actuator_profile)')
    before = '                        "actuator_contract.json": json.dumps(export_contract(actuator_profile)),'
    if expected.count(before) != 1:
        raise ValueError('Unknown TorchScript serialization location')
    expected = expected.replace(before,before+'\n                        "hip_contract.json": json.dumps(hip_contract()),')
    before = '        configure_hip(cfg)\n'
    if expected.count(before) != 1:
        raise ValueError('Unknown hip profile validation location')
    expected = expected.replace(before,
        '        if saved.get("hip_guard_profile") == HIP_PROFILE and saved.get("hip_contract") != hip_contract():\n'
        '            raise ValueError("Saved hip configuration differs from this continuation")\n'+before)
    if corrected != expected:
        raise ValueError('Correction contains an unqualified training/physics change')


def verify_save_expression(source):
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr == 'update' and isinstance(n.func.value,ast.Name) and n.func.value.id == 'saved'
             and any(k.arg == 'actuator_contract' for k in n.keywords)]
    if len(calls) != 1:
        raise ValueError('Unexpected checkpoint save expression')
    scope = dict(export_contract=export_contract,actuator_profile='rs0402_transient_v4',
                 HIP_PROFILE=PROFILE,hip_contract=contract)
    values = {k.arg:eval(compile(ast.Expression(k.value),'<actual-save-expression>','eval'),scope)
              for k in calls[0].keywords if k.arg in ('actuator_profile','actuator_contract','hip_guard_profile','hip_contract')}
    if checkpoint_profile(values) != 'rs0402_transient_v4' or values['hip_contract'] != contract():
        raise ValueError('Corrected metadata cannot round-trip')
    return values


def main():
    identity,_ = verify_prepared(HERE)
    _,old = verify_prepared(PREFLIGHT)
    current = json.loads((HERE/'PREPARED.json').read_text())
    if set(old['files']) != set(current['files']):
        raise ValueError('Runtime file set changed')
    changed = [n for n in old['files'] if old['files'][n] != current['files'][n]]
    if changed != ['train.py']:
        raise ValueError(f'Runtime beyond metadata changed: {changed}')
    corrected = (HERE/'code/train.py').read_text()
    prove_delta((PREFLIGHT/'code/train.py').read_text(),corrected)
    roundtrip = verify_save_expression(corrected)
    # The verified source predates V22, so the extra restore guard is inactive
    # during its 3000-step run; it rejects bad metadata on the later 16+984 resume.
    import torch
    saved = torch.load(identity['path'],map_location='cpu',weights_only=False)
    if saved.get('hip_guard_profile') is not None:
        raise ValueError('Bridge requires the exact pre-V22 source')
    if (PREFLIGHT/'VERIFIED_READY.json').exists():
        raise RuntimeError('Independent verifier was already invoked; inspect before retry')
    subprocess.run([sys.executable,str(PREFLIGHT/'verify_preflight.py')],check=True)
    ready = json.loads((PREFLIGHT/'VERIFIED_READY.json').read_text())
    if ready['source_sha256'] != identity['sha256'] or not ready['actual_runner_resume_verified']:
        raise ValueError('Preflight source mismatch')
    result = dict(state='verified_not_training',source_completed=34823,source_sha256=identity['sha256'],
        actual_runner_resume_verified=True,optimizer_updates=0,
        runtime_preflight_path=str(PREFLIGHT/'preflight_3000/FINALIZED.json'),
        runtime_preflight_sha256=hashlib.sha256((PREFLIGHT/'preflight_3000/FINALIZED.json').read_bytes()).hexdigest(),
        runtime_control_steps=3000,metadata_only_delta_proven=True,roundtrip_contract=roundtrip,
        corrected_entrypoint_has_not_run_3000_steps=True)
    with (HERE/'VERIFIED_READY.json').open('x') as f:
        json.dump(result,f,indent=2)
    print('VERIFIED unchanged3000-step physics + exact serialization correction and roundtrip')


if __name__ == '__main__':
    main()
