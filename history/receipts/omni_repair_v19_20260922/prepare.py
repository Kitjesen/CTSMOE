"""Freeze an isolated V19 runtime; never modify a running or historical version."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PARENT = Path('/root/autodl-tmp/cts-phase-pose-v18-20260922')
WEIGHTS = dict(pit=.10, stairs_up=.10, stairs_down=.08, wave=.08,
               slope_up=.06, slope_down=.06, rough_slope=.08, obstacles=.04, flat=.40)


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Frozen source mismatch: {old[:80]}')
    return text.replace(old, new, 1)


def main():
    target = HERE/'code'
    if target.exists():
        raise FileExistsError('V19 already prepared; preserve it')
    for name, expected in json.loads((PARENT/'PREPARED.json').read_text())['files'].items():
        assert hashlib.sha256((PARENT/'code'/name).read_bytes()).hexdigest() == expected, name
    shutil.copytree(PARENT/'code', target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('omni_commands.py', 'omni_lift.py', 'flat_boundary.py'):
        shutil.copyfile(HERE/name, target/name)
    path = target/'terrain_allocation.py'
    text = path.read_text()
    text = once(text, "EARLY_BALANCED_PROFILE = 'early_balanced_v14'", "EARLY_BALANCED_PROFILE = 'omni_repair_v19'")
    begin, end = text.index('EARLY_BALANCED = dict('), text.index('\nBASE = dict(')
    text = text[:begin] + 'EARLY_BALANCED = ' + repr(WEIGHTS) + text[end:]
    path.write_text(text)
    path = target/'train.py'
    text = path.read_text()
    text = once(text, '        configure_phase_pose(cfg)\n',
        '        configure_phase_pose(cfg)\n'
        '        from omni_commands import configure as configure_omni_commands\n'
        '        from omni_lift import configure as configure_omni_lift\n'
        '        configure_omni_commands(cfg)\n'
        '        configure_omni_lift(cfg)\n')
    text = once(text, '                saved.update(phase_pose_profile=cfg.phase_pose_profile,',
        '                saved.update(omni_repair_profile="omni_repair_v19", phase_pose_profile=cfg.phase_pose_profile,')
    # Checkpoints remain at 500-update intervals; initial 16 updates are an integration check.
    # Existing finite-loss/applied-torque safeguards remain active throughout.
    path.write_text(text)
    path = target/'early_restart.py'
    text = path.read_text()
    text = once(text, '    from phase_pose import snapshot as phase_pose_snapshot\n',
        '    from phase_pose import snapshot as phase_pose_snapshot\n'
        '    from omni_lift import snapshot as lift_snapshot\n')
    text = once(text, '        "phase_pose": phase_pose_snapshot(env),',
        '        "phase_pose": phase_pose_snapshot(env),\n'
        '        "omni_commands": command.omni_snapshot(),\n'
        '        "omni_lift": lift_snapshot(env),')
    text = once(text, '    terms = raw_terms(env)\n    gate, phase = phase_weight(env)\n',
        '    gate, phase = phase_weight(env)\n')
    text = once(text, '    sampling = (command.startup_sampling_snapshot()',
        '    terms = {key: env.reward_manager.get_term_cfg(name).func(\n'
        '        env, **env.reward_manager.get_term_cfg(name).params)\n'
        '        for key, name in term_names.items()}\n'
        '    sampling = (command.startup_sampling_snapshot()')
    text = once(text, '        "standing_multiplier": STANDING_SCALE,',
        '        "standing_multiplier": 1.0,\n'
        '        "reward_diagnostics_source": "actual active RewardTermCfg function and params",')
    path.write_text(text)
    receipt = dict(state='prepared_not_trained', source_completed=31831,
        parent_runtime=str(PARENT/'code'), num_envs=4096, new_update_budget=10000,
        allocation=WEIGHTS, intended_completed=41831,
        changes=['flat command coverage and full-episode stop-start sequences',
                 'V18 phase-scoped actual-angle L1 retained',
                 'flat lateral/yaw lift-and-landing feedback', 'fixed terrain proportions'],
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob('*.py')})
    (HERE/'PREPARED.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
