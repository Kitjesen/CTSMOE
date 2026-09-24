"""Build a new isolated runtime; never modify prior frozen experiments."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
FROZEN = Path('/root/cts-early-ablation-v15-20260921/code_train')
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Unexpected source revision: {old[:70]}')
    return text.replace(old, new, 1)


def fixed_commands(text):
    text = replace_once(text, 'self.commands[zero_env_ids, :2] = 0.0',
                        'self.commands[zero_env_ids, :] = 0.0')
    start = text.index('                if self.cfg.limit_ang_vel_at_zero_command_prob > 0.0:')
    end = text.index('            min_prob += self.zero_command_prob', start)
    # Full stop is a separate semantic category. Existing explicit yaw categories
    # remain in startup_sampling; a selected full stop must not turn into rotation.
    text = text[:start] + text[end:]
    return '# Thunder V17 fork: full-stop branch clears all three axes; see PREPARED.json.\n' + text


def main():
    out = HERE / 'code'
    if out.exists():
        raise FileExistsError('Preserve existing staged runtime')
    shutil.copytree(FROZEN, out, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    upstream = BASE / 'upstream/source/robot_lab/robot_lab/tasks/go2/mdp/commands.py'
    original = upstream.read_text()
    (out/'stop_commands.py').write_text(fixed_commands(original))
    path = out/'stair_speed.py'
    path.write_text(replace_once(path.read_text(),
        'from robot_lab.tasks.go2.mdp.commands import Go2RLGymCommand',
        'from stop_commands import Go2RLGymCommand'))
    shutil.copyfile(HERE/'posture_stop.py', out/'posture_stop.py')
    path = out/'train.py'
    text = path.read_text()
    text = replace_once(text, '        configure_early(cfg, early_profile)\n',
        '        configure_early(cfg, early_profile)\n'
        '        from posture_stop import configure as configure_posture_stop\n'
        '        if height_penalty != "original":\n'
        '            raise ValueError("V17 requires explicit --base-height-penalty original")\n'
        '        configure_posture_stop(cfg)\n')
    path.write_text(text)
    path = out/'early_restart.py'
    text = path.read_text()
    text = replace_once(text, 'def snapshot(env) -> dict | None:\n',
        'def snapshot(env) -> dict | None:\n    from posture_stop import snapshot as posture_stop_snapshot\n')
    text = replace_once(text, '        "profile": getattr(env.cfg, "early_restart_profile", PROFILE),',
        '        "posture_stop": posture_stop_snapshot(env),\n'
        '        "profile": getattr(env.cfg, "early_restart_profile", PROFILE),')
    path.write_text(text)
    receipt = dict(state='prepared_not_trained', source_completed=31831,
                   official_command_source=str(upstream),
                   official_command_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.py')})
    (HERE/'PREPARED.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
