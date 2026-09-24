"""Prepare an isolated default-only correction; preserve V22 and its evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'hip_guard_v22_20260923'
OUT = HERE / 'candidate'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one occurrence: {old[:100]}')
    return text.replace(old, new)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


shutil.copytree(OLD/'candidate_attempt03', OUT, ignore=shutil.ignore_patterns('__pycache__'))
for name in ('hip_contract.py', 'hip_guard.py'):
    shutil.copy2(HERE/name, OUT/'code'/name)
verify = (OUT/'verify_preflight.py').read_text()
verify = replace_once(verify, 'if max(final["hip_evidence"]["peak_inward_rad"]["target"]) > .120001:',
                      'if max(final["hip_evidence"]["target_abs_peak_rad"]) > .369001:')
verify = verify.replace('Wrong default or hip operating envelope', 'Wrong default-only hip contract')
verify = verify.replace('Hip target guard failed', 'Inherited hip soft limits failed')
(OUT/'verify_preflight.py').write_text(verify, encoding='utf-8')
runtime = (OUT/'runtime_contract.py').read_text().replace('V22', 'V23')
(OUT/'runtime_contract.py').write_text(runtime, encoding='utf-8')

shutil.copytree(OLD/'recovery01/assessment_v2', OUT/'assessment', ignore=shutil.ignore_patterns('__pycache__'))
for name in ('evaluate_hip.py', 'evaluate_hip_platform.py'):
    path = OUT/'assessment'/name
    text = path.read_text().replace('"v22"', '"v23"')
    path.write_text(text, encoding='utf-8')
path = OUT/'assessment/run_assessment.py'
path.write_text(path.read_text().replace('hip_inward012_v22', 'hip_default012_v23'), encoding='utf-8')
path = OUT/'assessment/hip_metrics.py'
metrics = path.read_text()
metrics = replace_once(metrics, "        i, count = case['id'], case['end_step']+1", """        terms = set(case.get('terms', []))
        known = {'illegal_contact', 'actuator_violation', 'pit_success', 'pit_outside', 'time_out', 'stair_route_exit'}
        if terms - known:
            raise ValueError(f'Unclassified termination reasons: {terms - known}')
        physical_failure = bool(terms & {'illegal_contact', 'actuator_violation'})
        i, count = case['id'], case['end_step']+1""")
metrics = metrics.replace("int(case.get('physical', False))", 'int(physical_failure)')
metrics = metrics.replace("physical=case.get('physical', False)", 'physical=physical_failure')
metrics = metrics.replace('seconds_inward_gt0121=', 'seconds_above_nominal_plus001_not_violation=')
metrics = metrics.replace("result = dict(fixed_cases_only=True", 'result = dict(default_is_not_a_limit=True, fixed_cases_only=True')
path.write_text(metrics, encoding='utf-8')

driver = (OLD/'recovery01/driver_v22_recovery01.py').read_text()
driver = driver.replace('V22', 'V23').replace('v22', 'v23').replace('hip_inward012_v23', 'hip_default012_v23')
driver = driver.replace('TARGET = 35823', 'TARGET = 35721\nDISCARDED_V22_UPDATES = 102')
driver = replace_once(driver, "    for name,digest in json.loads((HERE/'RECOVERY01_MANIFEST.json').read_text())['files'].items():\n        if file_hash(HERE/name) != digest:\n            raise RuntimeError(f'Frozen recovery changed: {name}')\n", '')
driver = driver.replace('completed-31831+8', 'completed-31831+8+DISCARDED_V22_UPDATES')
driver = driver.replace('STARTED_V23_RECOVERY01.json', 'STARTED_V23.json')
driver = driver.replace("        if not (ROOT/'hip_guard_v23_attempt02/READY_FOR_REVIEW.json').exists():\n            raise RuntimeError('Requires completed zero-update preflight')\n", '')
driver = driver.replace('max_added_updates=1000', 'max_added_updates=898, discarded_v22_updates=DISCARDED_V22_UPDATES')
start = driver.index('            # Completed independent qualification is retained')
end = driver.index('            for increment in (16,984):', start)
driver = driver[:start] + """            execute('qualification_driver', [sys.executable, str(HERE/'run_preflight.py')], completed)
            execute('independent_verification', [sys.executable, str(HERE/'verify_preflight.py')], completed)
            ready = json.loads((HERE/'VERIFIED_READY.json').read_text())
            if ready['state'] != 'verified_not_training' or ready['source_completed'] != completed:
                raise RuntimeError('Independent qualification did not pass')
            source = Path(identity['path'])
""" + driver[end:]
driver = driver.replace('for increment in (16,984):', 'for increment in (16,882):')
driver = driver.replace('assessment_v2/', 'assessment/').replace('evaluation_35823', 'evaluation_35721')
driver = driver.replace('source_new_hip:', 'source_default012:')
driver = driver.replace('qualified hip-envelope repair, 16+984 updates', 'default-only correction, 16+882 updates')
(OUT/'driver_v23.py').write_text(driver, encoding='utf-8')

prepared = json.loads((OUT/'PREPARED.json').read_text())
prepared['files'] = {p.relative_to(OUT/'code').as_posix(): sha(p) for p in (OUT/'code').rglob('*.py')}
prepared['correction'] = 'User requested default0.12 only; original hip target/soft/hard limits preserved'
(OUT/'PREPARED.json').write_text(json.dumps(prepared, indent=2), encoding='utf-8')
manifest = json.loads((OUT/'RUNTIME_PREPARE_MANIFEST.json').read_text())
manifest['files'] = {name: sha(OUT/name) for name in manifest['files']}
(OUT/'RUNTIME_PREPARE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
orchestration = {p.relative_to(OUT).as_posix(): sha(p) for p in (OUT/'assessment').glob('*.py')}
orchestration['driver_v23.py'] = sha(OUT/'driver_v23.py')
(OUT/'ORCHESTRATION_FILES.json').write_text(json.dumps(dict(files=orchestration), indent=2), encoding='utf-8')
with zipfile.ZipFile(HERE/'candidate_payload.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
    for p in OUT.rglob('*'):
        if p.is_file():
            archive.write(p, p.relative_to(OUT).as_posix())
print(sha(HERE/'candidate_payload.zip'))
