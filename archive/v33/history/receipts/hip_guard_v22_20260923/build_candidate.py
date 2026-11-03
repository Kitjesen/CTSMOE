"""Prepare an isolated V22 candidate from the preserved V21 source snapshot."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REF = HERE / 'reference_v21'
OUT = HERE / 'candidate'
REMOTE = '/root/autodl-tmp/cts-omni-repair-v19-20260922/hip_guard_v22'
ROOT = '/root/autodl-tmp/cts-omni-repair-v19-20260922'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Expected exactly one insertion location: {before[:100]}')
    return text.replace(before, after)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    shutil.copytree(REF / 'code', OUT / 'code')
    for name in ('hip_contract.py', 'hip_guard.py'):
        shutil.copy2(HERE / name, OUT / 'code' / name)
    train = (OUT / 'code/train.py').read_text(encoding='utf-8')
    train = replace_once(train, '        configure_omni_lift(cfg)\n',
        '        configure_omni_lift(cfg)\n'
        '        from hip_guard import configure as configure_hip, validate_runtime as validate_hip, HipEvidence\n'
        '        from hip_contract import PROFILE as HIP_PROFILE, contract as hip_contract\n'
        '        if saved.get("hip_guard_profile") not in (None, HIP_PROFILE):\n'
        '            raise ValueError("Unknown hip source profile")\n'
        '        configure_hip(cfg)\n')
    train = replace_once(train, '        terrain_cfg = cfg.scene.terrain.terrain_generator\n',
        '        validate_hip(raw)\n        hip_evidence = HipEvidence(raw)\n'
        '        terrain_cfg = cfg.scene.terrain.terrain_generator\n')
    train = replace_once(train, '                saved.update(omni_repair_profile=',
        '                saved.update(hip_guard_profile=HIP_PROFILE, hip_contract=hip_contract(), omni_repair_profile=')
    train = train.replace('export_contract(actuator_profile)',
                          '{**export_contract(actuator_profile), "hip_guard": hip_contract()}')
    train = replace_once(train, '"training_context.json": json.dumps({"wave_reset_profile": wave_profile,',
        '"training_context.json": json.dumps({"hip_guard_profile": HIP_PROFILE, "hip_contract": hip_contract(), "wave_reset_profile": wave_profile,')
    train = replace_once(train, '        run_receipt = dict(\n',
        '        run_receipt = dict(hip_guard_profile=HIP_PROFILE, hip_contract=validate_hip(raw),\n')
    train = replace_once(train, '            row = dict(state="training",',
        '            row = dict(hip_guard=hip_evidence.snapshot(), state="training",')
    train = replace_once(train, '            if (args.output / "STOP").exists():',
        '            hip_evidence.reset_statistics()\n            if (args.output / "STOP").exists():')
    (OUT / 'code/train.py').write_text(train, encoding='utf-8')

    identity = dict(path=ROOT+'/stair_coverage_v21/training_33881_to_34823/model_34822.pt',
        completed_updates=34823, common_step_counter=34823*24, bytes=27818003,
        sha256='521a31d1acfc7e0906f4ac3260ee30f161af5b1c18278367d0c0045b7afe3fa5',
        spent_new_updates=3000, remaining_new_updates=7000, discarded_optimizer_updates=8)
    (OUT / 'SOURCE_IDENTITY.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    for name in ('preflight_support.py', 'coverage_observation.py', 'run_preflight.py', 'verify_preflight.py'):
        text = (REF / name).read_text(encoding='utf-8')
        text = text.replace('source33865', 'source34823')
        (OUT / name).write_text(text, encoding='utf-8')
    pf = (REF / 'preflight.py').read_text(encoding='utf-8').replace('source33865', 'source34823')
    pf = replace_once(pf, '    raw = self.env.unwrapped\n',
        '    raw = self.env.unwrapped\n'
        '    from hip_guard import validate_runtime as validate_hip, HipEvidence\n'
        '    initial_hip = validate_hip(raw)\n'
        '    hip_evidence = HipEvidence(raw)\n')
    pf = replace_once(pf, "    (out/'FINALIZED.json').write_text(json.dumps(dict(state='complete',optimizer_updates=0,",
        "    validate_hip(raw)\n"
        "    hip_snapshot = hip_evidence.snapshot()\n"
        "    (out/'FINALIZED.json').write_text(json.dumps(dict(hip_contract=initial_hip, hip_evidence=hip_snapshot, state='complete',optimizer_updates=0,")
    (OUT / 'preflight.py').write_text(pf, encoding='utf-8')
    vf = (OUT / 'verify_preflight.py').read_text(encoding='utf-8')
    vf = replace_once(vf, '    initial = run[\'stair_transition_coverage\']\n',
        '    sys.path.insert(0, str(HERE / "code"))\n'
        '    from hip_contract import contract\n'
        '    if final["hip_contract"] != contract() or run["hip_contract"] != contract():\n'
        '        raise ValueError("Wrong default or hip operating envelope")\n'
        '    if max(final["hip_evidence"]["peak_inward_rad"]["target"]) > .120001:\n'
        '        raise ValueError("Hip target guard failed")\n'
        '    initial = run[\'stair_transition_coverage\']\n')
    (OUT / 'verify_preflight.py').write_text(vf, encoding='utf-8')
    runtime = (REF / 'runtime_contract.py').read_text(encoding='utf-8')
    runtime = runtime.replace('COMPLETED = 33865', 'COMPLETED = 34823').replace('COMMON_STEP = 812760', 'COMMON_STEP = 835752')
    runtime = runtime.replace('SOURCE_FILES = 86', 'SOURCE_FILES = 87')
    runtime = runtime.replace("'2666c590ac843ac2626a18286490fb44557cebdba6bfb1d3aaf7b639a373de62'", repr(sha(OUT / 'SOURCE_IDENTITY.json')))
    runtime = runtime.replace('speed_brake_v20/training_33174_to_34823/model_33864.pt',
                              'stair_coverage_v21/training_33881_to_34823/model_34822.pt')
    runtime = runtime.replace('Invalid preserved 33865 identity receipt', 'Invalid preserved 34823 identity receipt')
    begin, end = runtime.index('def verify_prepared(here):'), runtime.index('\ndef validate_coverage(receipt):')
    runtime = runtime[:begin] + '''def verify_prepared(here):
    here = Path(here)
    prepared = json.loads((here / 'PREPARED.json').read_text())
    added = {'hip_contract.py', 'hip_guard.py'}
    files, source = prepared['files'], prepared['source_files']
    if (len(source) != SOURCE_FILES or set(files) != set(source) | added
            or sorted(n for n in source if files[n] != source[n]) != ['train.py']):
        raise ValueError('V22 changes exceed explicit hip integration scope')
    if files['stair_transition_coverage.py'] != REVIEWED_COVERAGE_SHA256:
        raise ValueError('Inherited V21 geometry changed')
    verify_files(here / 'code', files, SOURCE_FILES + len(added))
    verify_files(Path(prepared['source_code']), source, SOURCE_FILES)
    for name, expected in json.loads((here / 'RUNTIME_PREPARE_MANIFEST.json').read_text())['files'].items():
        if file_sha256(here / name) != expected:
            raise ValueError(f'V22 qualification payload changed: {name}')
    identity = load_identity(here / IDENTITY_NAME)
    if prepared['source_identity'] != identity:
        raise ValueError('Mismatched V22 source identity')
    verify_source(identity)
    return identity, prepared

''' + runtime[end:]
    (OUT / 'runtime_contract.py').write_text(runtime, encoding='utf-8')
    prepared = dict(source_code=ROOT+'/stair_coverage_v21/code', source_identity=identity,
        source_files={p.relative_to(REF/'code').as_posix(): sha(p) for p in (REF/'code').rglob('*.py')},
        files={p.relative_to(OUT/'code').as_posix(): sha(p) for p in (OUT/'code').rglob('*.py')})
    (OUT / 'PREPARED.json').write_text(json.dumps(prepared, indent=2), encoding='utf-8')
    payload = {p.name: sha(p) for p in OUT.glob('*.py')}
    payload['SOURCE_IDENTITY.json'] = sha(OUT / 'SOURCE_IDENTITY.json')
    payload['PREPARED.json'] = sha(OUT / 'PREPARED.json')
    (OUT / 'RUNTIME_PREPARE_MANIFEST.json').write_text(json.dumps(dict(files=payload), indent=2), encoding='utf-8')
    print(json.dumps(dict(code_files=len(prepared['files']), source_files=len(prepared['source_files']),
                          source=identity, remote=REMOTE), indent=2))


if __name__ == '__main__':
    main()
