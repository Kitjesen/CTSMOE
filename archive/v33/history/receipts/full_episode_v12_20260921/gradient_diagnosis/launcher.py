"""Execute a 288-step V12 on-policy static-gradient diagnosis with zero updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback


EXPECTED_COMPLETED_UPDATES = 45011
EXPECTED_REFERENCE_COMPLETED_UPDATES = 43202
EXPECTED_FAST_DELAY = False
NUM_ENVS = 4096
HORIZONS = 12
STEPS_PER_HORIZON = 24
CONTROL_STEPS = HORIZONS*STEPS_PER_HORIZON
SMOKE_STEPS = STEPS_PER_HORIZON


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'code', 'output', 'checkpoint', 'reference'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_value(digest, value, torch):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode()+repr(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            hash_value(digest, key, torch)
            hash_value(digest, value[key], torch)
    elif isinstance(value, (list, tuple)):
        for item in value:
            hash_value(digest, item, torch)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode()+repr(value).encode())
    else:
        raise TypeError(type(value))


def state_sha256(state, torch):
    digest = hashlib.sha256()
    hash_value(digest, state, torch)
    return digest.hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def first_metric(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                return json.loads(line)
    raise ValueError(f'empty metrics receipt: {path}')


class MutationGuard:
    """Forbid training mutations while deliberately permitting compute_returns."""
    def __init__(self, runner):
        self.runner, self.calls, self.originals = runner, [], []
        self.restored = False

    def forbid(self, owner, name, label):
        original = getattr(owner, name)

        def blocked(*_args, **_kwargs):
            self.calls.append(label)
            raise RuntimeError(f'forbidden zero-update call: {label}')
        self.originals.append((owner, name, original))
        setattr(owner, name, blocked)

    def __enter__(self):
        alg = self.runner.alg
        try:
            self.forbid(alg, 'update', 'alg.update')
            self.forbid(alg.optimizer, 'step', 'ppo_optimizer.step')
            self.forbid(alg.optimizer_stu_enc, 'step', 'student_optimizer.step')
            self.forbid(self.runner, 'save', 'runner.save')
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.restored = all(getattr(owner, name) is original
                            for owner, name, original in self.originals)

    def __exit__(self, *_args):
        self.restore()


def allocate_diagnostic_storage(alg, torch):
    shape = (alg.storage.num_transitions_per_env, alg.anchor_env.num_envs)
    device = alg.device
    specs = {
        'audit_age_s': (1, torch.float32), 'audit_env_id': (1, torch.long),
        'audit_group': (1, torch.long), 'audit_command': (3, torch.float32),
        'audit_flat_cohort': (1, torch.bool), 'audit_position': (12, torch.float32),
        'audit_position_target': (12, torch.float32),
        'audit_position_request': (12, torch.float32),
        'audit_wheel_velocity': (4, torch.float32),
        'audit_wheel_velocity_target': (4, torch.float32),
        'audit_support_count': (1, torch.long), 'audit_base_contact': (1, torch.bool),
    }
    for name, (width, dtype) in specs.items():
        alg.storage.observations[name] = torch.zeros(*shape, width, device=device, dtype=dtype)


def augment_observations(obs, alg, raw, torch):
    command = raw.command_manager.get_term('base_velocity')
    robot = raw.scene['robot'].data
    pit = command.pit
    result = obs.clone(recurse=False)
    count = raw.num_envs
    group = torch.zeros(count, device=raw.device, dtype=torch.long)
    group[alg.student_env_idxs] = 1
    forces = torch.stack([sensor.data.force_matrix_w[:, 0, 0].norm(dim=-1)
                          for sensor in pit.ground_contacts], dim=-1)
    base_force = pit.contact.data.net_forces_w_history[:, :, pit.base_contact_ids].norm(
        dim=-1).amax(dim=(1, 2))
    position_term = raw.action_manager.get_term('joint_pos')
    velocity_term = raw.action_manager.get_term('joint_vel')
    position_ids = position_term.joint_indices
    # Deployed IsaacLab JointAction resolves native JointVelocityAction joints
    # into _joint_ids (joint_actions.py:65, consumed at :246/:250).
    wheel_ids = velocity_term._joint_ids
    result['audit_age_s'] = (raw.episode_length_buf*raw.step_dt)[:, None]
    result['audit_env_id'] = torch.arange(count, device=raw.device)[:, None]
    result['audit_group'] = group[:, None]
    result['audit_command'] = command.command.clone()
    result['audit_flat_cohort'] = ((command.terrain_idxs == command.terrain_type2idx['flat'])
                                   & alg.anchor_cohort)[:, None]
    result['audit_position'] = robot.joint_pos[:, position_ids].clone()
    result['audit_position_target'] = robot.joint_pos_target[:, position_ids].clone()
    result['audit_position_request'] = position_term.requested_actions.clone()
    result['audit_wheel_velocity'] = robot.joint_vel[:, wheel_ids].clone()
    result['audit_wheel_velocity_target'] = robot.joint_vel_target[:, wheel_ids].clone()
    result['audit_support_count'] = (forces > 5.).sum(-1, keepdim=True)
    result['audit_base_contact'] = (base_force > 5.)[:, None]
    expected = {
        'audit_position': (count, 12), 'audit_position_target': (count, 12),
        'audit_position_request': (count, 12), 'audit_wheel_velocity': (count, 4),
        'audit_wheel_velocity_target': (count, 4),
    }
    for name, shape in expected.items():
        if result[name].shape != shape:
            raise ValueError(f'{name} has shape {result[name].shape}, expected {shape}')
    return result


def first_native_minibatch(alg, torch, seed):
    device = torch.device(alg.device)
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        return next(alg.storage.mini_batch_generator(alg.num_mini_batches, 1))


def make_diagnostic_learn(torch, output, smoke):
    total_steps = SMOKE_STEPS if smoke else CONTROL_STEPS

    def diagnostic_learn(self, num_learning_iterations, init_at_random_ep_len=False):
        status_path = output/'audit_status.json'
        before, guard = {}, None
        try:
            if num_learning_iterations != 0 or self.current_learning_iteration != EXPECTED_COMPLETED_UPDATES:
                raise RuntimeError('diagnosis requires V12 completed_updates=45011 and --updates 0')
            if init_at_random_ep_len:
                raise RuntimeError('diagnosis requires episode age from zero')
            if self.env.num_envs != NUM_ENVS or self.cfg['num_steps_per_env'] != STEPS_PER_HORIZON:
                raise RuntimeError('diagnosis requires 4096 envs and native 24-step storage')
            alg, raw = self.alg, self.env.unwrapped
            policy = alg.policy
            if policy.actor_obs_normalization or policy.critic_obs_normalization:
                raise RuntimeError('diagnosis requires the saved no-normalization V12 path')
            if alg.rnd is not None:
                raise RuntimeError('diagnosis scope excludes unregistered RND gradients')
            if alg.storage.step != 0:
                raise RuntimeError('rollout storage not empty')
            before = {
                'model': state_sha256(alg.policy.state_dict(), torch),
                'reference': state_sha256(alg.anchor_reference.state_dict(), torch),
                'ppo_optimizer': state_sha256(alg.optimizer.state_dict(), torch),
                'student_optimizer': state_sha256(alg.optimizer_stu_enc.state_dict(), torch),
                'coefficients': json.dumps(alg.anchor_coefficients, sort_keys=True),
            }
            allocate_diagnostic_storage(alg, torch)
            from ledger import GradientLedger
            ledger = GradientLedger(alg, raw, output)
            obs = self.env.get_observations().to(self.device)
            self.train_mode()
            guard = MutationGuard(self)
            collected = 0
            with guard:
                horizon = 0
                while collected < total_steps:
                    steps = min(STEPS_PER_HORIZON, total_steps-collected)
                    with torch.inference_mode():
                        for _ in range(steps):
                            augmented = augment_observations(obs, alg, raw, torch)
                            actions = alg.act(augmented)
                            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                            obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                            alg.process_env_step(obs, rewards, dones, extras)
                            collected += 1
                        alg.compute_returns(obs)
                    batch = first_native_minibatch(alg, torch, 45011+horizon)
                    device = torch.device(alg.device)
                    devices = [device.index] if device.type == 'cuda' else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(55011+horizon)
                        if device.type == 'cuda':
                            torch.cuda.manual_seed_all(55011+horizon)
                        ledger.analyze(batch, horizon)
                    if any(parameter.grad is not None for parameter in alg.policy.parameters()):
                        raise RuntimeError('static diagnosis accumulated parameter.grad')
                    alg.storage.clear()
                    horizon += 1
            summary = ledger.finish()
            after = {
                'model': state_sha256(alg.policy.state_dict(), torch),
                'reference': state_sha256(alg.anchor_reference.state_dict(), torch),
                'ppo_optimizer': state_sha256(alg.optimizer.state_dict(), torch),
                'student_optimizer': state_sha256(alg.optimizer_stu_enc.state_dict(), torch),
                'coefficients': json.dumps(alg.anchor_coefficients, sort_keys=True),
            }
            if before != after or guard.calls or not guard.restored:
                raise RuntimeError('zero-update mutation invariant failed')
            write_json(status_path, {
                'state': 'DIAGNOSIS_SMOKE_COMPLETE' if smoke else 'DIAGNOSIS_COMPLETE',
                'authoritative': True, 'smoke_only': smoke,
                'completed_updates': EXPECTED_COMPLETED_UPDATES, 'optimizer_updates': 0,
                'control_steps': collected, 'horizons': summary['horizons'],
                'num_envs': NUM_ENVS, 'state_hashes_before': before, 'state_hashes_after': after,
                'fast_delay': EXPECTED_FAST_DELAY,
                'forbidden_calls': [], 'compute_returns_used': True,
                'fresh_on_policy_advantages': True, 'checkpoint_artifacts_written': [],
            })
        except BaseException as exc:
            write_json(status_path, {
                'state': 'DIAGNOSIS_FAILED', 'optimizer_updates': 0, 'error': str(exc),
                'traceback': traceback.format_exc(), 'state_hashes_before': before,
                'forbidden_calls': [] if guard is None else guard.calls,
            })
            raise
    return diagnostic_learn


def main():
    args = parse_args()
    root, code, output = args.root.resolve(), args.code.resolve(), args.output.resolve()
    checkpoint, reference = args.checkpoint.resolve(), args.reference.resolve()
    if output.exists():
        raise FileExistsError(output)
    metrics = checkpoint.parent/'metrics.jsonl'
    for path in (code/'train.py', code/'anchor_cts.py', code/'anchor_objective.py',
                 checkpoint, reference, metrics, Path(__file__).with_name('ledger.py')):
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path[:0] = [str(Path(__file__).parent), str(code), str(root/'upstream/source/rsl_rl')]
    import torch
    from rsl_rl.runners import OnPolicyRunnerCTS

    checkpoint_state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    reference_state = torch.load(reference, map_location='cpu', weights_only=False)
    if checkpoint_state['completed_updates'] != EXPECTED_COMPLETED_UPDATES:
        raise ValueError('checkpoint is not V12 completed_updates=45011')
    if checkpoint_state['startup_repair_profile'] != 'omni_full_episode_v1':
        raise ValueError('checkpoint is not the full-episode profile')
    if reference_state['completed_updates'] != EXPECTED_REFERENCE_COMPLETED_UPDATES:
        raise ValueError('reference is not V6 completed_updates=43202')
    if first_metric(metrics).get('fast_delay') is not EXPECTED_FAST_DELAY:
        raise ValueError('diagnosis must inherit fast_delay=false from the V12 pilot')
    checkpoint_hash, reference_hash = file_sha256(checkpoint), file_sha256(reference)
    original_learn, original_argv = OnPolicyRunnerCTS.learn, sys.argv[:]
    OnPolicyRunnerCTS.learn = make_diagnostic_learn(torch, output, args.smoke)
    sys.argv = [str(code/'train.py'), '--root', str(root), '--output', str(output),
                '--num-envs', str(NUM_ENVS), '--updates', '0', '--mixed-pit',
                '--resume', str(checkpoint), '--startup-repair-profile', 'omni_full_episode_v1',
                '--startup-anchor-reference', str(reference)]
    try:
        runpy.run_path(str(code/'train.py'), run_name='__main__')
    finally:
        sys.argv = original_argv
        OnPolicyRunnerCTS.learn = original_learn
    status_path = output/'audit_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    expected = 'DIAGNOSIS_SMOKE_COMPLETE' if args.smoke else 'DIAGNOSIS_COMPLETE'
    if status.get('state') != expected:
        raise RuntimeError(status)
    artifacts = sorted(output.glob('model_*.pt'))+sorted(output.glob('student_*.pt'))
    if artifacts:
        raise RuntimeError(f'unexpected checkpoint artifacts: {artifacts}')
    status.update({
        'checkpoint': str(checkpoint), 'reference': str(reference),
        'checkpoint_sha256': checkpoint_hash, 'reference_sha256': reference_hash,
        'input_files_unchanged': (file_sha256(checkpoint) == checkpoint_hash
                                  and file_sha256(reference) == reference_hash),
        'launcher_patch_restored': OnPolicyRunnerCTS.learn is original_learn,
        'checkpoint_artifacts_written': [], 'training_success_receipt_superseded': True,
    })
    write_json(status_path, status)
    print(json.dumps({'state': expected, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
