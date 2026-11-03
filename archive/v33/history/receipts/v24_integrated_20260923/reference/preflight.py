"""V21 geometry-only runtime qualification: 3000 control steps, zero optimizer updates."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'code'))
from runtime_contract import (COMMON_STEP, COMPLETED, IDENTITY_NAME, NUM_ENVS, STEPS,  # noqa: E402
                              load_identity, validate_coverage, verify_prepared)

IDENTITY = load_identity(HERE / IDENTITY_NAME)
import early_restart  # noqa: E402

original_configure = early_restart.configure


def configure(cfg, profile):
    original_configure(cfg, profile)
    from rsl_rl.runners import OnPolicyRunnerCTS
    OnPolicyRunnerCTS.learn = preflight


def preflight(self, updates, init_at_random_ep_len=False):
    import torch
    from preflight_support import MutationGuard, state_sha256
    from posture_stop import snapshot as base_snapshot
    from phase_pose import snapshot as pose_snapshot
    from omni_lift import snapshot as lift_snapshot
    def snapshot(env):
        result = base_snapshot(env)
        result["phase_pose"] = pose_snapshot(env)
        result["omni"] = env.command_manager.get_term("base_velocity").omni_snapshot()
        result["lift"] = lift_snapshot(env)
        result["allocation"] = env.terrain_allocation.snapshot()
        names = ('landing_motion', 'lin_vel_z_l2', 'normal_vertical_velocity', 'early_vertical_velocity',
                 'early_actual_hip_l1', 'early_actual_thigh_calf_l1', 'base_height_l2', 'joint_pos_limits')
        result['effective_reward_audit'] = {name: {
            'registered': name in env.reward_manager.active_terms,
            'weight': env.reward_manager.get_term_cfg(name).weight if name in env.reward_manager.active_terms else None}
            for name in names}
        for name in ("early_actual_hip_l1", "early_actual_thigh_calf_l1"):
            assert env.reward_manager.get_term_cfg(name).weight == -.20
        assert result["phase_pose"]["by_terrain"]["flat"]["gate_min"] == 1.
        return result
    if (updates != 0 or init_at_random_ep_len or self.env.num_envs != NUM_ENVS
            or self.current_learning_iteration != COMPLETED
            or self.cfg['num_steps_per_env'] != 24 or self.alg.storage.step != 0):
        raise ValueError('Requires source34823, 4096 environments, empty 24-step storage, and zero updates')
    verify_prepared(HERE)
    out = Path(sys.argv[sys.argv.index('--output')+1])
    source = Path(sys.argv[sys.argv.index('--resume')+1])
    def fingerprint():
        return [state_sha256(v,torch) for v in (self.alg.policy.state_dict(), self.alg.optimizer.state_dict(),
                                               self.alg.optimizer_stu_enc.state_dict())]
    before = fingerprint()
    if source.resolve() != Path(IDENTITY['path']).resolve():
        raise ValueError('Wrong preflight checkpoint path')
    run = json.loads((out/'run.json').read_text())
    if not all(run.get('resume_checks', {}).get(name) is True for name in (
            'model', 'ppo_optimizer', 'student_optimizer', 'terrain_assignment',
            'command_terrain_membership', 'terrain_restored_exactly', 'terrain_allocation',
            'stair_episode_totals')):
        raise ValueError('Source state/assignment restoration was not exact')
    geometry = json.loads((out/'COVERAGE_GEOMETRY.json').read_text())
    validate_coverage(geometry)
    if geometry != run['stair_transition_coverage']:
        raise ValueError('Coverage geometry receipt differs from restored run snapshot')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != IDENTITY['sha256']:
        raise ValueError('Source checkpoint changed')
    raw = self.env.unwrapped
    from hip_guard import validate_runtime as validate_hip, HipEvidence
    initial_hip = validate_hip(raw)
    hip_evidence = HipEvidence(raw)
    if raw.common_step_counter != COMMON_STEP:
        raise ValueError('Source common-step clock was not restored')
    command = raw.command_manager.get_term('base_velocity')
    from coverage_observation import CoverageObservation
    coverage_observation = CoverageObservation(raw, torch)
    if not any(c.__module__ == 'stop_commands' for c in command.__class__.__mro__):
        raise RuntimeError('Corrected full-stop sampler is not in active command hierarchy')
    obs = self.env.get_observations().to(self.device)
    self.train_mode()
    guard = MutationGuard(self)
    full_zero_steps = rotation_steps = transition_to_stop = transition_from_stop = 0
    phase_coverage = {name: [1., 0.] for name in ("flat", "stairs", "pit")}
    observed = torch.zeros(3, device=raw.device)
    previous_stop = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)
    with guard, torch.inference_mode():
        for step in range(STEPS):
            action = self.alg.act(obs)
            obs, reward, done, extras = self.env.step(action.to(raw.device))
            obs, reward, done = obs.to(self.device), reward.to(self.device), done.to(self.device)
            self.alg.process_env_step(obs,reward,done,extras)
            coverage_observation.observe(raw, done.to(raw.device))
            if not all(bool(torch.isfinite(v).all()) for v in (action,reward,raw.scene['robot'].data.joint_pos)):
                raise FloatingPointError('Nonfinite runtime state')
            if bool(raw.actuator_guard.violation_causes['torque'].any()):
                raise RuntimeError('Applied torque limit violated')
            flat = command.terrain_idxs == command.terrain_type2idx['flat']
            observed = torch.maximum(observed, command.command[flat].abs().amax(0))
            stop = flat & (command.command == 0).all(-1)
            same_episode = raw.episode_length_buf > 1
            full_zero_steps += int(stop.sum())
            rotation_steps += int((flat & (command.command[:,:2] == 0).all(-1) & (command.command[:,2] != 0)).sum())
            transition_to_stop += int((stop & ~previous_stop & same_episode).sum())
            transition_from_stop += int((flat & ~stop & previous_stop & same_episode).sum())
            previous_stop = stop.clone()
            if (step+1)%24 == 0:
                self.alg.storage.clear()
                receipt = snapshot(raw)
                for name, value in receipt["phase_pose"]["by_terrain"].items():
                    phase_coverage[name][0] = min(phase_coverage[name][0], value["gate_min"])
                    phase_coverage[name][1] = max(phase_coverage[name][1], value["gate_max"])
            if step%300 == 0:
                print('PREFLIGHT',step,full_zero_steps,rotation_steps,flush=True)
        self.alg.storage.clear()
    after = fingerprint()
    if before != after or guard.calls or not guard.restored:
        raise RuntimeError('Preflight mutated policy or optimizers')
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source checkpoint changed')
    if min(full_zero_steps,rotation_steps,transition_to_stop,transition_from_stop) <= 0:
        raise RuntimeError('Missing real full-stop/rotation/stop-start sampling coverage')
    if (raw.common_step_counter != COMMON_STEP + STEPS
            or self.current_learning_iteration != COMPLETED or list(out.rglob('*.pt'))):
        raise RuntimeError('Preflight advanced optimizer age or wrote checkpoint artifacts')
    from stair_transition_coverage import snapshot as coverage_snapshot
    end_coverage = coverage_snapshot(raw)
    validate_coverage(end_coverage)
    if end_coverage['mesh_receipts'] != geometry['mesh_receipts']:
        raise RuntimeError('Terrain geometry changed during preflight')
    receipt['stair_transition_coverage'] = end_coverage
    verify_prepared(HERE)
    assert phase_coverage["flat"] == [1., 1.]
    assert bool((observed >= torch.tensor([1.9, .95, 1.4], device=raw.device)).all())
    assert bool((observed <= torch.tensor([2.00001, 1.00001, 1.50001], device=raw.device)).all())
    assert receipt["omni"]["boundary"]["crossings"] == 0
    assert receipt["lift"]["landing_events_since_env_init"] > 0
    assert all(value['tracking_gate_env_steps'] > 0 for name, value in receipt['lift']['cumulative_by_command'].items() if name != 'other')
    assert all(min(row) > 0 for row in receipt["omni"]["kind_counts_by_learner"])
    assert all(min(row) > 0 for row in receipt["omni"]["stop_restart_by_learner"])
    assert receipt["allocation"]["pending_moves"] == 0
    for name, weight in receipt["allocation"]["target_weights"].items():
        assert abs(receipt["allocation"]["actual_counts"][name] - 4096*weight) <= 3
    assert phase_coverage["pit"][0] < .1 and phase_coverage["pit"][1] > .9
    assert phase_coverage["stairs"][0] < .1 and phase_coverage["stairs"][1] > .9
    validate_hip(raw)
    hip_snapshot = hip_evidence.snapshot()
    (out/'FINALIZED.json').write_text(json.dumps(dict(hip_contract=initial_hip, hip_evidence=hip_snapshot, state='complete',optimizer_updates=0,
        steps=STEPS, num_envs=NUM_ENVS, source_completed=COMPLETED, checkpoint_sha256=source_hash,
        common_step_before=COMMON_STEP, common_step_after=raw.common_step_counter,
        runner_iteration_before=COMPLETED, runner_iteration_after=self.current_learning_iteration,
        state_hashes_before=dict(zip(('model', 'ppo_optimizer', 'student_optimizer'), before)),
        state_hashes_after=dict(zip(('model', 'ppo_optimizer', 'student_optimizer'), after)),
        forbidden_calls=guard.calls, mutation_guard_restored=guard.restored,
        checkpoint_artifacts_written=[], initial_coverage=geometry,
        coverage_observation=coverage_observation.snapshot(),
        observed_flat_command_abs_max=observed.tolist(),
        full_zero_flat_env_steps=full_zero_steps,rotation_flat_env_steps=rotation_steps,
        within_episode_stop_transitions=transition_to_stop,within_episode_restart_transitions=transition_from_stop,
        settings=receipt, phase_coverage=phase_coverage, qualification='Runtime/sampling validation, not learned posture improvement'),indent=2))


early_restart.configure = configure
if __name__ == '__main__':
    if sys.flags.optimize:
        raise RuntimeError('Preflight assertions must remain enabled')
    verify_prepared(HERE)
    from train import main
    main()
    out = Path(sys.argv[sys.argv.index('--output')+1])
    receipt = json.loads((out/'FINALIZED.json').read_text())
    if receipt['optimizer_updates'] != 0 or receipt['source_completed'] != COMPLETED:
        raise RuntimeError('The zero-update learn replacement did not complete')
    verify_prepared(HERE)
    for name in ('SUCCESS.json','status.json'):
        (out/name).write_text(json.dumps(dict(state='preflight_complete',completed_updates=COMPLETED,optimizer_updates=0)))
