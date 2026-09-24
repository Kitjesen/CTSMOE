"""Build V21 from the actual audited V20 runtime; never launch Isaac or training."""
import argparse
import ast
import json
from pathlib import Path
import shutil

from runtime_contract import (
    IDENTITY_NAME, REVIEWED_COVERAGE_SHA256, ROOT, SOURCE_FILES, SOURCE_RELATIVE, file_sha256,
    load_identity, verify_files, verify_source,
)

HERE = Path(__file__).resolve().parent
PAYLOAD_FILES = (
    'build_runtime.py', 'runtime_contract.py', 'preflight.py', 'run_preflight.py',
    'verify_preflight.py', 'preflight_support.py', 'coverage_observation.py', IDENTITY_NAME,
)
CONFIG_ANCHOR = '            configure_mixed(cfg, slope_profile=slope_profile, pit_training_profile=pit_profile)\n'
CONFIG_INSERT = (
    '            from stair_transition_coverage import configure as configure_stair_transition_coverage\n'
    '            configure_stair_transition_coverage(cfg)\n'
)
RECEIPT_ANCHOR = '        run_receipt = dict(\n'
SNAPSHOT_INSERT = (
    '        from stair_transition_coverage import snapshot as stair_transition_coverage_snapshot\n'
    '        coverage_geometry = stair_transition_coverage_snapshot(raw)\n'
    '        (args.output / "COVERAGE_GEOMETRY.json").write_text(json.dumps(coverage_geometry, indent=2))\n'
)
RECEIPT_INSERT = '            stair_transition_coverage=coverage_geometry,\n'


def patch_train(text):
    if 'stair_transition_coverage' in text:
        raise ValueError('Train already contains V21 coverage')
    for fragment in (CONFIG_ANCHOR, RECEIPT_ANCHOR):
        if text.count(fragment) != 1:
            raise ValueError(f'Expected one original train insertion anchor: {fragment.strip()}')
    patched = text.replace(CONFIG_ANCHOR, CONFIG_ANCHOR + CONFIG_INSERT)
    patched = patched.replace(RECEIPT_ANCHOR, SNAPSHOT_INSERT + RECEIPT_ANCHOR + RECEIPT_INSERT)
    # Fail closed if the original restoration/order changes upstream.
    if not (patched.index(CONFIG_INSERT) < patched.index('raw = Go2Env(cfg=cfg)')
            < patched.index('resume_checks["terrain_allocation"] = True')
            < patched.index(SNAPSHOT_INSERT) < patched.index('del saved')):
        raise ValueError('Coverage insertion would precede runtime construction or restoration')
    recovered = patched.replace(CONFIG_INSERT, '').replace(SNAPSHOT_INSERT, '').replace(RECEIPT_INSERT, '')
    if recovered != text:
        raise ValueError('Train transformation exceeds the three documented insertions')
    ast.parse(patched)
    return patched


def build(root, payload, coverage_module):
    root, payload, coverage_module = Path(root).resolve(), Path(payload).resolve(), Path(coverage_module).resolve()
    destination = root / 'stair_coverage_v21'
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError('V21 destination must be absent or strictly empty; preserve existing evidence')
    if destination == payload or destination in payload.parents:
        raise ValueError('Payload must be staged outside the destination')
    identity = load_identity(payload / IDENTITY_NAME)
    if Path(identity['path']).resolve() != (root / SOURCE_RELATIVE).resolve():
        raise ValueError('Identity source path does not match the selected frozen ROOT')
    verify_source(identity)
    original = root / 'speed_brake_v20'
    source_files = json.loads((original / 'PREPARED.json').read_text(encoding='utf-8'))['files']
    verify_files(original / 'code', source_files, SOURCE_FILES)
    if coverage_module.name != 'stair_transition_coverage.py' or not coverage_module.is_file():
        raise ValueError('Expected reviewed stair_transition_coverage.py payload')
    if file_sha256(coverage_module) != REVIEWED_COVERAGE_SHA256:
        raise ValueError('Coverage module differs from the explicitly reviewed version')
    for name in PAYLOAD_FILES:
        if not (payload / name).is_file():
            raise FileNotFoundError(payload / name)
    patched = patch_train((original / 'code/train.py').read_text(encoding='utf-8'))
    ast.parse(coverage_module.read_text(encoding='utf-8'))
    # Every check that can be made before creation is above; a partial build is preserved, never replaced.
    destination.mkdir(exist_ok=True)
    code = destination / 'code'
    code.mkdir()
    for name in source_files:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / 'code' / name, target)
    (code / 'train.py').write_text(patched, encoding='utf-8', newline='\n')
    shutil.copyfile(coverage_module, code / coverage_module.name)
    for name in PAYLOAD_FILES:
        shutil.copyfile(payload / name, destination / name)
    files = {name: file_sha256(code / name) for name in [*source_files, coverage_module.name]}
    changed = sorted(name for name in source_files if files[name] != source_files[name])
    if changed != ['train.py']:
        raise ValueError(f'Unexpected source changes: {changed}')
    verify_files(code, files, SOURCE_FILES + 1)
    verify_files(original / 'code', source_files, SOURCE_FILES)
    prepared = dict(state='prepared_not_started', source_identity=identity,
                    source_code=str(original / 'code'), source_file_count=SOURCE_FILES,
                    source_files=source_files, files=files, changed_source_files=changed,
                    added_runtime_files=[coverage_module.name],
                    train_insertions=['configure immediately after configure_mixed',
                                      'snapshot after restored assignments, courses and learner groups',
                                      'same snapshot in COVERAGE_GEOMETRY.json and run.json'],
                    optimizer_updates=0, launched=False)
    (destination / 'PREPARED.json').write_text(json.dumps(prepared, indent=2), encoding='utf-8')
    payload_hashes = {name: file_sha256(destination / name) for name in PAYLOAD_FILES}
    (destination / 'RUNTIME_PREPARE_MANIFEST.json').write_text(
        json.dumps(dict(files=payload_hashes), indent=2), encoding='utf-8')
    print(json.dumps(dict(state='prepared_not_started', destination=str(destination),
                          source_files=SOURCE_FILES, runtime_files=len(files), changed=changed)))
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--payload-dir', type=Path, default=HERE)
    parser.add_argument('--coverage-module', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.payload_dir, args.coverage_module)


if __name__ == '__main__':
    main()
