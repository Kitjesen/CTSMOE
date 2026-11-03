"""Extend the frozen 19538 teacher and critic without changing deployed student weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from thunder_lab3.tasks.thunder.mdp.privileged import FEATURE_NAMES

SOURCE_SHA256 = '8db552f69ab35105f737e74be9477c82e2d539a5d3984b2c32e41c1e992c226e'
WEIGHTS = (('teacher_encoder.0.network.0.weight', 274, 0), ('critic.network.0.weight', 306, 1))


def extend_columns(value, columns):
    if not isinstance(value, torch.Tensor) or value.shape != (512, columns):
        raise ValueError('Unexpected teacher/critic weight or Adam moment shape')
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite source weight or moment')
    return torch.cat((value, value.new_zeros((512, 36))), dim=1)


def migrate_checkpoint(saved):
    expected = {'frame': 53, 'history': 10, 'critic': 274, 'actions': 16, 'experts': 8, 'latent': 32}
    if saved.get('completed_updates') != 19538 or any(saved['signature'].get(k) != v for k, v in expected.items()):
        raise ValueError('Expected the frozen 19538 observation/checkpoint contract')
    if any('normalizer' in name for name in saved['model']):
        raise ValueError('Observation-normalizer state requires a separate migration')
    groups = saved['ppo_optimizer']['param_groups']
    if len(groups) != 4 or [len(g['params']) for g in groups] != [6, 8, 8, 1]:
        raise ValueError('Unexpected PPO parameter layout')
    ids = [p for g in groups for p in g['params']]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated optimizer parameter IDs')
    result = copy.deepcopy(saved)
    for name, columns, group in WEIGHTS:
        result['model'][name] = extend_columns(saved['model'][name], columns)
        pid = groups[group]['params'][0]
        state = result['ppo_optimizer']['state'][pid]
        for field, value in state.items():
            if isinstance(value, torch.Tensor) and value.ndim and field not in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                raise ValueError('Unknown nonscalar optimizer state: ' + field)
        for field in ('exp_avg', 'exp_avg_sq'):
            state[field] = extend_columns(state[field], columns)
        if 'max_exp_avg_sq' in state:
            state['max_exp_avg_sq'] = extend_columns(state['max_exp_avg_sq'], columns)
    result['signature']['critic'] = 310
    result['signature']['privileged_extension'] = list(FEATURE_NAMES)
    result['privileged_migration'] = {'version': 1, 'source_updates': 19538,
        'expected_source_sha256': SOURCE_SHA256, 'source_hash_verified': False, 'physics_state_continuity': False,
        'note': 'Zero new columns and Adam moments; student, shared actor and student optimizer preserved.'}
    return result


def migrate_newton_checkpoint(saved):
    """Newton has one mu and restitution: explicitly drop the old dynamic-friction input.

    The removed column changes teacher/critic predictions. This is a warm start,
    not a forward-equivalent migration or a seamless continuation of dynamics.
    """
    result = migrate_checkpoint(saved)
    for name, _, group in WEIGHTS:
        removed = 272 if group == 0 else 32 + 272
        def drop(value):
            return torch.cat((value[:, :removed], value[:, removed+1:]), dim=1)
        result['model'][name] = drop(result['model'][name])
        pid = result['ppo_optimizer']['param_groups'][group]['params'][0]
        state = result['ppo_optimizer']['state'][pid]
        for field in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if field in state:
                state[field] = drop(state[field])
    result['signature']['critic'] = 309
    result['signature']['material_schema'] = ['newton_shared_mu', 'restitution']
    result['privileged_migration'].update(version=2, removed_old_critic_feature_index=272,
        teacher_critic_forward_preserved=False,
        note='Dynamic-friction feature/weight/Adam column removed; static-friction column reused for Newton shared mu. '
             'New privileged columns zero; student, shared actor and student optimizer preserved.')
    return result


def migrate_file(source, output, newton=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source:
        raise ValueError('Output must be a new checkpoint file')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Source checksum differs from verified 19538 checkpoint')
    saved = torch.load(source, map_location='cpu', weights_only=False)
    migrated = migrate_newton_checkpoint(saved) if newton else migrate_checkpoint(saved)
    migrated['privileged_migration'].update(source_sha256=digest, source_hash_verified=True)
    with output.open('xb') as stream:
        torch.save(migrated, stream)
    loaded = torch.load(output, map_location='cpu', weights_only=False)
    for name, value in migrated['model'].items():
        if not torch.equal(value, loaded['model'][name]):
            raise ValueError('Serialized model changed')
    return {'source_sha256': digest, 'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'completed_updates': loaded['completed_updates'], 'teacher_dim': loaded['signature']['critic'], 'student_frame': 53,
            'teacher_critic_forward_preserved': not newton}


def restore_teacher_checkpoint(policy, algorithm, saved, newton=False):
    """Call after constructing ActorCriticMoECTS from a 310-D critic observation.

    Returns the migrated checkpoint for existing terrain/RNG restoration code.
    It does not reset the environment or claim cross-backend trajectory continuity.
    """
    if saved['signature'].get('critic') == 274:
        saved = migrate_newton_checkpoint(saved) if newton else migrate_checkpoint(saved)
    expected_dim = 309 if newton else 310
    if saved['signature'].get('critic') != expected_dim or saved['signature'].get('privileged_extension') != list(FEATURE_NAMES):
        raise ValueError('Unknown extended teacher observation layout')
    if newton and saved['signature'].get('material_schema') != ['newton_shared_mu', 'restitution']:
        raise ValueError('Unknown Newton material layout')
    if not 19538 <= saved['completed_updates'] <= 20000:
        raise ValueError('Checkpoint update count outside the continuation contract')
    policy.load_state_dict(saved['model'], strict=True)
    algorithm.optimizer.load_state_dict(saved['ppo_optimizer'])
    algorithm.optimizer_stu_enc.load_state_dict(saved['student_optimizer'])
    algorithm.learning_rate = saved['learning_rate']
    for name, _, group in WEIGHTS:
        parameter = algorithm.optimizer.param_groups[group]['params'][0]
        if parameter is not dict(policy.named_parameters())[name]:
            raise ValueError('PPO optimizer maps to a different teacher/critic parameter')
        for field in ('exp_avg', 'exp_avg_sq'):
            if algorithm.optimizer.state[parameter][field].shape != parameter.shape:
                raise ValueError('Restored Adam moment does not match extended parameter')
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('output')
    parser.add_argument('--newton', action='store_true')
    args = parser.parse_args()
    print(json.dumps(migrate_file(args.source, args.output, args.newton), indent=2))
