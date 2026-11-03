import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name):
    if sys.platform == 'win32':
        sys.modules.setdefault('fcntl', SimpleNamespace())
    spec = importlib.util.spec_from_file_location('v20_'+name, HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preserved_budget_exact_handoff_and_fixed_targets():
    driver = load('driver')
    verifier = load('verify_preflight')
    assert driver.SOURCE.as_posix() == verifier.SOURCE_PATH
    assert 'training_32831_to_34823/model_33173.pt' in verifier.SOURCE_PATH
    assert driver.SOURCE_SHA == verifier.SOURCE_SHA
    assert driver.budget(33174)['spent_new_updates'] == 1351
    assert 33174 + sum(driver.REMAINING_STAGES) == 41823
    assert driver.budget(41823)['remaining_new_updates'] == 0
    with pytest.raises(ValueError):
        driver.budget(41824)


def test_full_continuation_gate_function_is_unchanged():
    old = HERE.parent/'omni_repair_v19_20260922/recovery01/resume_after_verify.py'
    def node(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verdict_for'))
    assert node(old) == node(HERE/'driver.py')


def fixture():
    old = HERE.parent/'omni_repair_v19_20260922/recovery01'
    final = json.loads((old/'PREFLIGHT_FINALIZED.json').read_text(encoding='utf-8'))
    run = json.loads((old/'PREFLIGHT_RUN.json').read_text(encoding='utf-8'))
    v = load('verify_preflight')
    final.update(source_completed=33174, checkpoint_sha256=v.SOURCE_SHA,
                 observed_flat_command_abs_max=[1.99, .999, 1.499])
    final['settings']['omni']['profile'] = 'omni_full_episode_v20'
    s = final['settings']
    s['height_course_current_step'] = s['height_course_applied_step'] = 796176+3000
    s['height_weight'] = -1-9*min(max((s['height_course_applied_step']//24-31831)/5000, 0), 1)
    run.update(resume=v.SOURCE_PATH, resume_sha256=v.SOURCE_SHA, starting_completed_updates=33174)
    return v, final, run, 'RESUME_OK 33174\n', dict(completed_updates=33174, common_step_counter=796176)


def test_independent_verifier_checks_updated_identity_and_range():
    v, final, run, log, saved = fixture()
    v.validate(final, run, log, saved)
    final['observed_flat_command_abs_max'][0] = 1.
    with pytest.raises(AssertionError):
        v.validate(final, run, log, saved)


def test_verifier_rejects_wrong_source_and_optimizer_restore():
    v, final, run, log, saved = fixture()
    run['resume'] = run['resume'].replace('training_32831_', 'training_33174_')
    with pytest.raises(ValueError):
        v.validate(final, run, log, saved)
    run['resume'] = v.SOURCE_PATH
    run['resume_checks']['student_optimizer'] = False
    with pytest.raises(ValueError):
        v.validate(final, run, log, saved)


def test_braking_segments_cover_stop_reverse_and_restart():
    f = load('focus_protocol')
    assert f.brake_factor(torch.tensor([0., 4.99, 5., 9.99, 10., 14.99, 15., 19.99, 20., 24.99])).tolist() == [1.,1.,0.,0.,-1.,-1.,0.,0.,1.,1.]


def test_focus_cases_preserve_policy_contiguity_and_independent_modes():
    f = load('focus_protocol')
    def builder(selection, suite):
        if suite == 'flat25':
            return [dict(id=i*2+j, policy=i, mode=mode, terrain='flat', level=0)
                    for i in range(len(selection)) for j, mode in enumerate(['direct', 'settle'])]
        return [dict(id=i*2+j, policy=i, mode='direct', terrain=terrain, level=9)
                for i in range(len(selection)) for j, terrain in enumerate(['stairs_up', 'stairs_down'])]
    cases = f.build_focus_cases([0,1,2,3], 'brake25', builder)
    assert len(cases) == 144
    assert [c['policy'] for c in cases] == sorted(c['policy'] for c in cases)
    assert [c['id'] for c in cases] == list(range(144))
    stairs = f.build_focus_cases([0,1,2,3], 'stairend', builder)
    assert len(stairs) == 16
    assert {c['landing_mode'] for c in stairs} == {'continue', 'stop'}
