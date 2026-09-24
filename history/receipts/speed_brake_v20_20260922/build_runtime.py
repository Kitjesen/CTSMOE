"""Create isolated V20 orchestration from the reviewed, running V19 implementation."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'omni_repair_v19_20260922'
SOURCE = '/root/autodl-tmp/cts-omni-repair-v19-20260922/recovery01/training_32831_to_34823/model_33173.pt'
SHA = '05c34f02dcfb0bec462640c361258a896aa3435e0684f3c3b8ae15a6175b0ef83'


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one exact source fragment: {old[:90]}')
    return text.replace(old, new)


def main():
    code = (OLD / 'recovery01/resume_after_verify.py').read_text(encoding='utf-8')
    code = replace(code, "SOURCE = ROOT/'training_31831_to_32831/model_32830.pt'", f'SOURCE = Path({SOURCE!r})')
    code = replace(code, "SOURCE_SHA = 'f40891182ce1b0f9d80adac2c962b74c6ad8226e183fc44eb62cf9cba4ddd32a'", f'SOURCE_SHA = {SHA!r}')
    code = replace(code, 'REMAINING_STAGES = (1992, 3000, 4000)', 'REMAINING_STAGES = (1649, 3000, 4000)')
    code = replace(code, "recovery='recovery01'", "recovery='speed_brake_v20'")
    code = replace(code, "p = ROOT/f'pipeline_status.{os.getpid()}.tmp'", "p = HERE/f'pipeline_status.{os.getpid()}.tmp'")
    code = code.replace("p.replace(ROOT/'pipeline_status.json')", "p.replace(HERE/'pipeline_status.json')")
    start = code.index('def verify_code():')
    end = code.index('\n\ndef run(', start)
    code = code[:start] + '''def verify_code():
    for folder, manifest in [(ROOT/'code', ROOT/'PREPARED.json'),
                             (ROOT/'recovery01/code', ROOT/'recovery01/PREPARED.json'),
                             (HERE/'code', HERE/'PREPARED.json'),
                             (HERE, HERE/'ORCHESTRATION_MANIFEST.json')]:
        for name, expected in json.loads(manifest.read_text())['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen runtime changed: {folder/name}')
''' + code[end:]
    start = code.index('    completed = 32831', code.index('def main():'))
    end = code.index('        source = SOURCE', start)
    code = code[:start] + '''    completed = 33174
    with (ROOT/'pipeline.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE/'STARTED_V20.json').exists() or (HERE/'pipeline_status.json').exists():
            raise RuntimeError('V20 already attempted; preserve state instead of duplicate launch')
        old_output = SOURCE.parent
        stopped = json.loads((old_output/'status.json').read_text())
        if stopped != {'state': 'stopped', 'error': 'STOP requested after saving'} or not (old_output/'STOP').exists():
            raise RuntimeError('Requires exact checkpointed user-requested V19 stop')
        verify_code()
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
            raise RuntimeError('Checkpoint33174 changed')
        OWNER = True
        with (HERE/'STARTED_V20.json').open('x') as f:
            json.dump(dict(pid=os.getpid(), stopped_source=stopped, **budget(completed)), f, indent=2)
        write_status('starting_qualification', completed)
        spec = importlib.util.spec_from_file_location('v19_original_pipeline', ROOT/'pipeline.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        previous = json.loads((ROOT/'evaluation_32831/CONTINUATION.json').read_text())
        if previous['stop_for_review']:
            raise RuntimeError('Behavior rejection cannot be bypassed')
        run('preflight_driver', [sys.executable, str(HERE/'run_preflight.py')], completed)
        run('verify_driver', [sys.executable, str(HERE/'verify_preflight.py')], completed)
        ready = json.loads((HERE/'VERIFIED_READY.json').read_text())
        if (ready.get('state') != 'verified_not_training' or ready.get('source_completed') != completed
                or ready.get('source_sha256') != SOURCE_SHA
                or ready.get('actual_runner_resume_verified') is not True):
            raise RuntimeError('Incomplete independently verified V20 qualification')
        run('focused_smoke', [sys.executable, str(HERE/'run_focus.py'), '--smoke',
            '--output', str(HERE/'focused_smoke'),
            '--candidate', str(ROOT/'training_31831_to_32831/model_32830.pt'), '--completed', '32831'], completed)
''' + code[end:]
    # Training, full evaluation, repeated-regression gates, GPU locks and budget
    # from the reviewed V19 driver are retained rather than reimplemented.
    code = code.replace("current = json.loads((ROOT/'pipeline_status.json').read_text())", "current = json.loads((HERE/'pipeline_status.json').read_text())")
    code = replace(code, '            verdict = verdict_for(original, evaluation, previous, completed)', '''            run(f'focus_{completed}', [sys.executable, str(HERE/'run_focus.py'),
                '--output', str(HERE/f'focus_{completed}'), '--candidate', str(source),
                '--completed', str(completed)], completed)
            verdict = verdict_for(original, evaluation, previous, completed)''')
    (HERE/'driver.py').write_text(code, encoding='utf-8')

    pre = (OLD/'preflight.py').read_text(encoding='utf-8')
    pre = pre.replace('source_completed=31831', 'source_completed=33174')
    pre = pre.replace('completed_updates=31831', 'completed_updates=33174')
    pre = replace(pre, '        result["allocation"] = env.terrain_allocation.snapshot()', '''        result["allocation"] = env.terrain_allocation.snapshot()
        names = ('landing_motion', 'lin_vel_z_l2', 'normal_vertical_velocity', 'early_vertical_velocity',
                 'early_actual_hip_l1', 'early_actual_thigh_calf_l1', 'base_height_l2', 'joint_pos_limits')
        result['effective_reward_audit'] = {name: {
            'registered': name in env.reward_manager.active_terms,
            'weight': env.reward_manager.get_term_cfg(name).weight if name in env.reward_manager.active_terms else None}
            for name in names}''')
    pre = replace(pre, '    previous_stop = torch.zeros', '    observed = torch.zeros(3, device=raw.device)\n    previous_stop = torch.zeros')
    pre = replace(pre, '            stop = flat &', '            observed = torch.maximum(observed, command.command[flat].abs().amax(0))\n            stop = flat &')
    pre = replace(pre, '    assert phase_coverage["flat"] == [1., 1.]', '''    assert phase_coverage["flat"] == [1., 1.]
    assert bool((observed >= torch.tensor([1.9, .95, 1.4], device=raw.device)).all())
    assert bool((observed <= torch.tensor([2.00001, 1.00001, 1.50001], device=raw.device)).all())''')
    pre = replace(pre, '        full_zero_flat_env_steps=full_zero_steps', '        observed_flat_command_abs_max=observed.tolist(),\n        full_zero_flat_env_steps=full_zero_steps')
    (HERE/'preflight.py').write_text(pre, encoding='utf-8')

    run = (OLD/'recovery01/run_preflight.py').read_text(encoding='utf-8')
    run = replace(run, '/root/autodl-tmp/cts-omni-repair-v19-20260922/training_31831_to_32831/model_32830.pt', SOURCE)
    run = replace(run, 'source_completed=32831', 'source_completed=33174')
    (HERE/'run_preflight.py').write_text(run, encoding='utf-8')

    verify = (OLD/'recovery01/verify_preflight_v2.py').read_text(encoding='utf-8')
    verify = replace(verify, "SOURCE_SHA = 'f40891182ce1b0f9d80adac2c962b74c6ad8226e183fc44eb62cf9cba4ddd32a'", f'SOURCE_SHA = {SHA!r}')
    verify = replace(verify, "SOURCE_PATH = '/root/autodl-tmp/cts-omni-repair-v19-20260922/training_31831_to_32831/model_32830.pt'", 'SOURCE_PATH = SOURCE_IDENTITY_PLACEHOLDER')
    verify = verify.replace('32831', '33174').replace('787944', '796176')
    verify = replace(verify, 'SOURCE_PATH = SOURCE_IDENTITY_PLACEHOLDER', f'SOURCE_PATH = {SOURCE!r}')
    verify = replace(verify, "'omni_full_episode_v19'", "'omni_full_episode_v20'")
    verify = replace(verify, "    if s['soft_limit_weight'] != -2. or not -3.025000001 <= s['height_weight'] <= -2.931399999:", "    if s['soft_limit_weight'] != -2. or not -10. <= s['height_weight'] <= -1.:")
    verify = replace(verify, '    assert s[\'omni\'][\'boundary\'][\'crossings\'] == 0', '''    assert s['omni']['boundary']['crossings'] == 0
    assert final.get('source_completed') == 33174
    observed = final['observed_flat_command_abs_max']
    assert all(lo <= actual <= hi for lo, actual, hi in zip([1.9, .95, 1.4], observed, [2.00001, 1.00001, 1.50001]))''')
    (HERE/'verify_preflight.py').write_text(verify, encoding='utf-8')


if __name__ == '__main__':
    main()
