"""Deploy only into a new isolated directory; preserve every V19 file."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922')
DEST = ROOT/'speed_brake_v20'
ARCHIVE = ROOT/'speed_brake_v20_deploy.zip'


def main():
    if DEST.exists():
        raise RuntimeError('Deployment directory already exists; do not overwrite')
    original_manifest = json.loads((ROOT/'recovery01/PREPARED.json').read_text())['files']
    for name, sha in original_manifest.items():
        if hashlib.sha256((ROOT/'recovery01/code'/name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Original recovery code changed')
    DEST.mkdir()
    shutil.copytree(ROOT/'recovery01/code', DEST/'code')
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or path.suffix not in ('.py', '.md', '.json'):
                raise ValueError(f'Unexpected deployment member {name}')
        archive.extractall(DEST)
    for name in ('omni_commands.py', 'omni_lift.py'):
        shutil.copyfile(DEST/'command_patch'/name, DEST/'code'/name)
    shutil.copyfile(DEST/'focus_patch/focus_metrics.py', DEST/'focus_metrics.py')
    files = {name: hashlib.sha256((DEST/'code'/name).read_bytes()).hexdigest() for name in original_manifest}
    # Only the two explicitly authorized runtime changes; no reward-weight,
    # actuator, checkpoint restore, or training-loop alteration is included.
    changed = sorted(name for name in files if files[name] != original_manifest[name])
    if changed != ['omni_commands.py', 'omni_lift.py']:
        raise RuntimeError(f'Unexpected runtime difference {changed}')
    (DEST/'PREPARED.json').write_text(json.dumps(dict(files=files, changed_runtime_files=changed), indent=2))
    orchestration = ('driver.py', 'preflight.py', 'run_preflight.py', 'verify_preflight.py',
                     'run_focus.py', 'evaluate_focus.py', 'focus_protocol.py', 'focus_metrics.py')
    hashes = {name: hashlib.sha256((DEST/name).read_bytes()).hexdigest() for name in orchestration}
    (DEST/'ORCHESTRATION_MANIFEST.json').write_text(json.dumps(dict(files=hashes), indent=2))
    print(json.dumps(dict(state='prepared_not_started', changed_runtime_files=changed, frozen_files=len(files))))


if __name__ == '__main__':
    main()
