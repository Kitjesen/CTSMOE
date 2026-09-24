"""Exercise actual upstream sampling logic and support-height gating without Isaac."""
import ast
from pathlib import Path
from types import SimpleNamespace, ModuleType
from typing import Sequence
import sys

import pytest
import torch

from prepare import fixed_commands
from posture_stop import height_cost, height_weight
from verify_preflight import validate, SOURCE_PATH, SOURCE_SHA


def sampler(source):
    tree = ast.parse(source)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Go2RLGymCommand')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_resample')
    namespace = dict(torch=torch, Sequence=Sequence,
                     sample_single_interval=lambda ids, lo, hi, device: torch.full((len(ids),), .4))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual sampling method>', 'exec'), namespace)
    return namespace['_resample']


def fake_command(probability):
    count = 128
    return SimpleNamespace(
        _env=SimpleNamespace(max_episode_length_s=25., episode_length_buf=torch.zeros(count), step_dt=.02),
        cfg=SimpleNamespace(command_range_curriculum=[], resampling_time=5., dynamic_resample_commands=False,
                            limit_vel_prob=0., zero_command_curriculum={}, limit_ang_vel_at_zero_command_prob=.2),
        terrain_length=8., max_lin_vel=1., device='cpu',
        env_command_ranges={k:torch.tensor([[-1., 1.]]).repeat(count,1)
                            for k in ('lin_vel_x','lin_vel_y','ang_vel_yaw')},
        commands_xy_accumulation=torch.zeros(count,2), time_left=torch.zeros(count),
        commands=torch.zeros(count,3), get_current_scale=lambda cfg: probability)


def test_original_bug_and_repaired_full_stop():
    source = (Path(__file__).parent/'upstream_commands.py').read_text(encoding='utf-8')
    for text, expected_stop in ((source, False), (fixed_commands(source), True)):
        command = fake_command(1.)
        sampler(text)(command, torch.arange(128))
        assert bool((command.commands == 0).all()) is expected_stop
        assert bool((command.commands[:, :2] == 0).all())


def test_motion_and_duration_remain_unchanged():
    source = (Path(__file__).parent/'upstream_commands.py').read_text(encoding='utf-8')
    command = fake_command(0.)
    sampler(fixed_commands(source))(command, torch.arange(128))
    torch.testing.assert_close(command.commands, torch.full((128,3), .4))
    torch.testing.assert_close(command.time_left, torch.full((128,), 5.))


def test_course_uses_restart_age_not_old_absolute_age():
    assert height_weight(31831) == -1.
    assert height_weight(32831) == pytest.approx(-2.8)
    assert height_weight(36831) == -10.
    assert height_weight(50000) == -10.


def test_height_is_released_only_on_relevant_geometry(monkeypatch):
    module = ModuleType('robot_lab.tasks.go2.mdp.rewards')
    module.base_height_l2 = lambda *args: torch.tensor([.01, .02, .03, .04])
    monkeypatch.setitem(sys.modules, module.__name__, module)
    phase = ModuleType('phase_transition')
    phase.measure = lambda env: dict(relevant=torch.tensor([False,True,True,True]),
        ground=torch.zeros(4), weight=torch.tensor([1.,0.,.5,1.]))
    monkeypatch.setitem(sys.modules, 'phase_transition', phase)
    env = SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([[0.,0.,.56]]).repeat(4,1)))})
    torch.testing.assert_close(height_cost(env,.46,None),torch.tensor([.01,0.,.005,.01]))
    env.scene['robot'].data.root_pos_w[3,2] = float('nan')
    with pytest.raises(FloatingPointError):
        height_cost(env,.46,None)


def test_completion_rejects_wrong_runner_and_checkpoint():
    final = dict(state='complete',steps=3000,num_envs=4096,optimizer_updates=0,
                 checkpoint_sha256=SOURCE_SHA,settings=dict(soft_limit_weight=-2.,height_weight=-1.225),
                 full_zero_flat_env_steps=1,rotation_flat_env_steps=1,
                 within_episode_stop_transitions=1,within_episode_restart_transitions=1)
    run = dict(resume=SOURCE_PATH,resume_sha256=SOURCE_SHA,starting_completed_updates=31831,additional_updates=0,
               resume_checks=dict(model=True,ppo_optimizer=True,student_optimizer=True))
    saved = dict(completed_updates=31831,common_step_counter=763944)
    validate(final,run,'RESUME_OK 31831\n',saved)
    with pytest.raises(ValueError,match='runner iteration'):
        validate(final,run,'RESUME_OK 33034\n',saved)
    with pytest.raises(ValueError,match='checkpoint hash'):
        validate(dict(final,checkpoint_sha256='wrong'),run,'RESUME_OK 31831\n',saved)
