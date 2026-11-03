"""Stage V18 as an isolated copy of the completed V17 runtime."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PREVIOUS = Path('/root/autodl-tmp/cts-posture-stop-v17-20260922')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Unexpected frozen source: {old[:90]}')
    return text.replace(old, new, 1)


def main():
    target = HERE / 'code'
    if target.exists():
        raise FileExistsError('V18 runtime already exists')
    old_manifest = json.loads((PREVIOUS / 'PREPARED.json').read_text())
    for name, expected in old_manifest['files'].items():
        assert hashlib.sha256((PREVIOUS / 'code' / name).read_bytes()).hexdigest() == expected, name
    shutil.copytree(PREVIOUS / 'code', target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(HERE / 'phase_pose.py', target / 'phase_pose.py')
    path = target / 'train.py'
    text = path.read_text()
    text = replace_once(text, '        configure_posture_stop(cfg)\n',
        '        configure_posture_stop(cfg)\n'
        '        from phase_pose import configure as configure_phase_pose\n'
        '        configure_phase_pose(cfg)\n')
    text = replace_once(text, '                saved.update(completed_updates=it + 1,',
        '                saved.update(phase_pose_profile=cfg.phase_pose_profile, completed_updates=it + 1,')
    path.write_text(text)
    path = target / 'early_restart.py'
    text = path.read_text()
    text = replace_once(text, '    from posture_stop import snapshot as posture_stop_snapshot\n',
        '    from posture_stop import snapshot as posture_stop_snapshot\n'
        '    from phase_pose import snapshot as phase_pose_snapshot\n')
    text = replace_once(text, '        "posture_stop": posture_stop_snapshot(env),',
        '        "posture_stop": posture_stop_snapshot(env),\n'
        '        "phase_pose": phase_pose_snapshot(env),')
    path.write_text(text)
    receipt = dict(state='prepared_not_trained', source_completed=31831,
        parent_runtime=str(PREVIOUS / 'code'),
        only_training_objective_change='extra actual-angle L1, phase scoped; no new observation/action constraints',
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob('*.py')})
    (HERE / 'PREPARED.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
