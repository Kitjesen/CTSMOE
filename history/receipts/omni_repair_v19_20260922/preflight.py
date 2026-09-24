"""Zero-update runtime qualification for the three requested V17 fixes."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'code'))
sys.path.append('/root/autodl-tmp/cts-uphill-runup-v16-20260922')
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
        for name in ("early_actual_hip_l1", "early_actual_thigh_calf_l1"):
            assert env.reward_manager.get_term_cfg(name).weight == -.20
        assert result["phase_pose"]["by_terrain"]["flat"]["gate_min"] == 1.
        return result
    if updates != 0 or init_at_random_ep_len or self.env.num_envs != 4096:
        raise ValueError('Requires 4096 environments and zero updates')
    out = Path(sys.argv[sys.argv.index('--output')+1])
    source = Path(sys.argv[sys.argv.index('--resume')+1])
    def fingerprint():
        return [state_sha256(v,torch) for v in (self.alg.policy.state_dict(), self.alg.optimizer.state_dict(),
                                               self.alg.optimizer_stu_enc.state_dict())]
    before = fingerprint()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = self.env.unwrapped
    command = raw.command_manager.get_term('base_velocity')
    if not any(c.__module__ == 'stop_commands' for c in command.__class__.__mro__):
        raise RuntimeError('Corrected full-stop sampler is not in active command hierarchy')
    obs = self.env.get_observations().to(self.device)
    self.train_mode()
    guard = MutationGuard(self)
    full_zero_steps = rotation_steps = transition_to_stop = transition_from_stop = 0
    phase_coverage = {name: [1., 0.] for name in ("flat", "stairs", "pit")}
    previous_stop = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)
    with guard, torch.inference_mode():
        for step in range(3000):
            action = self.alg.act(obs)
            obs, reward, done, extras = self.env.step(action.to(raw.device))
            obs, reward, done = obs.to(self.device), reward.to(self.device), done.to(self.device)
            self.alg.process_env_step(obs,reward,done,extras)
            if not all(bool(torch.isfinite(v).all()) for v in (action,reward,raw.scene['robot'].data.joint_pos)):
                raise FloatingPointError('Nonfinite runtime state')
            if bool(raw.actuator_guard.violation_causes['torque'].any()):
                raise RuntimeError('Applied torque limit violated')
            flat = command.terrain_idxs == command.terrain_type2idx['flat']
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
    if before != fingerprint() or guard.calls or not guard.restored:
        raise RuntimeError('Preflight mutated policy or optimizers')
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source checkpoint changed')
    if min(full_zero_steps,rotation_steps,transition_to_stop,transition_from_stop) <= 0:
        raise RuntimeError('Missing real full-stop/rotation/stop-start sampling coverage')
    assert phase_coverage["flat"] == [1., 1.]
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
    (out/'FINALIZED.json').write_text(json.dumps(dict(state='complete',optimizer_updates=0,
        steps=3000, num_envs=4096, source_completed=31831, checkpoint_sha256=source_hash,
        full_zero_flat_env_steps=full_zero_steps,rotation_flat_env_steps=rotation_steps,
        within_episode_stop_transitions=transition_to_stop,within_episode_restart_transitions=transition_from_stop,
        settings=receipt, phase_coverage=phase_coverage, qualification='Runtime/sampling validation, not learned posture improvement'),indent=2))


early_restart.configure = configure
if __name__ == '__main__':
    from train import main
    main()
    out = Path(sys.argv[sys.argv.index('--output')+1])
    receipt = json.loads((out/'FINALIZED.json').read_text())
    for name in ('SUCCESS.json','status.json'):
        (out/name).write_text(json.dumps(dict(state='preflight_complete',completed_updates=31831,optimizer_updates=0)))
