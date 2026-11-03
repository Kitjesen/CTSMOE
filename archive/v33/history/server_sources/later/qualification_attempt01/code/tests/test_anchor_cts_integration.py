"""Exercise the actual upstream CTS storage and both optimizer paths on CPU."""
import copy
import hashlib
from types import SimpleNamespace

import torch
import pytest
from tensordict import TensorDict
from rsl_rl.algorithms import MoECTS
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageCTS

from anchor_cts import AnchoredMoECTS
import anchor_cts
from anchor_objective import deterministic_mean


def setup_pair(tmp_path, monkeypatch, enabled=True):
    torch.manual_seed(321)
    obs = TensorDict({'policy': torch.randn(8, 20), 'single_obs': torch.randn(8, 5),
                     'critic': torch.randn(8, 12)}, batch_size=[8])
    policy = ActorCriticMoECTS(obs, {'policy': ['policy'], 'critic': ['critic']}, 16,
        actor_hidden_dims=[16], critic_hidden_dims=[16], teacher_encoder_hidden_dims=[16],
        student_encoder_hidden_dims=[16], expert_num=2, latent_dim=8)
    reference = copy.deepcopy(policy)
    with torch.no_grad():
        for param in reference.actor.parameters():
            param.add_(.01)
    ref_path = tmp_path/'reference.pt'
    torch.save(dict(model_state_dict=reference.state_dict(), completed_updates=43202), ref_path)
    monkeypatch.setattr(anchor_cts, 'REFERENCE_SHA256', hashlib.sha256(ref_path.read_bytes()).hexdigest())
    kwargs = dict(num_envs=8, num_learning_epochs=1, num_mini_batches=2, schedule='fixed')
    anchor = AnchoredMoECTS(copy.deepcopy(policy), RolloutStorageCTS('rl', 8, 6, 4, obs, [16]), **kwargs)
    plain = MoECTS(copy.deepcopy(policy), RolloutStorageCTS('rl', 8, 6, 4, obs, [16]), **kwargs)
    command = SimpleNamespace(terrain_idxs=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
                             terrain_type2idx={'flat': 0}, command=torch.zeros(8, 3))
    terms = {'joint_pos': SimpleNamespace(_scale=.25, action_dim=12),
             'joint_vel': SimpleNamespace(_scale=5., action_dim=4)}
    env = SimpleNamespace(num_envs=8, step_dt=.02, episode_length_buf=torch.full((8,), 0 if enabled else 100),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        action_manager=SimpleNamespace(active_terms=list(terms), get_term=lambda name: terms[name]))
    anchor.attach_anchor(env, ref_path)
    return obs, anchor, plain


def rollout(alg, obs):
    torch.manual_seed(53)
    with torch.no_grad():
        for _ in range(4):
            alg.act(obs)
            alg.process_env_step(obs, torch.linspace(.1, .8, 8), torch.zeros(8), {})
        alg.compute_returns(obs)


def test_training_mask_and_reference_follow_cts_shuffle(tmp_path, monkeypatch):
    obs, alg, _ = setup_pair(tmp_path, monkeypatch)
    rollout(alg, obs)
    for batch in alg.storage.mini_batch_generator(2, 1):
        grouped = batch[0]
        with torch.no_grad():
            expected = deterministic_mean(alg.anchor_reference, grouped, is_teacher=False)
        mask = grouped['anchor_mask'].squeeze(-1)
        assert mask.any() and (~mask).any()
        torch.testing.assert_close(grouped['anchor_reference_action'][mask], expected[mask])
    reference_before = copy.deepcopy(alg.anchor_reference.state_dict())
    result = alg.update()
    assert result['anchor_main'] > 0 and result['anchor_student'] > 0
    assert result['anchor_main_coefficient'] > 0 and result['anchor_student_coefficient'] > 0
    torch.testing.assert_close(alg.anchor_reference.state_dict(), reference_before, atol=0, rtol=0)
    assert not any('anchor' in k for k in alg.policy.state_dict())
    ppo_ids = {id(p) for g in alg.optimizer.param_groups for p in g['params']}
    student_ids = {id(p) for g in alg.optimizer_stu_enc.param_groups for p in g['params']}
    assert not (ppo_ids & student_ids)


def test_no_mask_exactly_preserves_upstream_update(tmp_path, monkeypatch):
    obs, anchor, plain = setup_pair(tmp_path, monkeypatch, enabled=False)
    rollout(anchor, obs)
    rollout(plain, obs)
    torch.manual_seed(77)
    anchor.update()
    torch.manual_seed(77)
    plain.update()
    torch.testing.assert_close(anchor.policy.state_dict(), plain.policy.state_dict(), rtol=0, atol=0)
    assert anchor.anchor_coefficients == dict(main=None, student=None)
    anchor.anchor_env.episode_length_buf.zero_()
    rollout(anchor, obs)
    anchor.update()
    assert all(v is not None and v > 0 for v in anchor.anchor_coefficients.values())


@pytest.mark.parametrize('invalid', [0., -.1, float('nan'), float('inf'), 1.01, True])
def test_resume_rejects_invalid_coefficients(tmp_path, monkeypatch, invalid):
    _, anchor, _ = setup_pair(tmp_path, monkeypatch)
    state = copy.deepcopy(anchor.anchor_state())
    state['coefficients']['main'] = invalid
    with pytest.raises(ValueError, match='coefficients'):
        anchor.attach_anchor(anchor.anchor_env, tmp_path/'reference.pt', state)


def test_resume_restores_calibration_and_coefficients(tmp_path, monkeypatch):
    obs, anchor, _ = setup_pair(tmp_path, monkeypatch)
    rollout(anchor, obs)
    anchor.update()
    state = copy.deepcopy(anchor.anchor_state())
    for calibration in state['calibration'].values():
        assert 0 < calibration['achieved_ratio'] <= .1000001
    # Clear cached Normal distribution before deepcopy: real runners construct
    # a fresh policy before loading a checkpoint and attaching the reference.
    anchor.policy.distribution = None
    anchor.attach_anchor(anchor.anchor_env, tmp_path/'reference.pt', state)
    assert anchor.anchor_state()['coefficients'] == state['coefficients']
    assert anchor.anchor_state()['calibration'] == state['calibration']
