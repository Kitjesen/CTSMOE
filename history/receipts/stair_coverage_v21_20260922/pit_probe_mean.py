"""Read-only-policy probe of the actual mixed training pit acceptance path."""
import hashlib
import json
import math
from pathlib import Path
import sys

V21 = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922/stair_coverage_v21')
sys.path[:0] = [str(V21/'code'), str(V21)]
import early_restart  # noqa: E402

STEPS = 750
COMPLETED = 34823
ORIGINAL_CONFIGURE = early_restart.configure


def configure(cfg, profile):
    ORIGINAL_CONFIGURE(cfg, profile)
    from rsl_rl.runners import OnPolicyRunnerCTS
    OnPolicyRunnerCTS.learn = probe


def probe(self, updates, init_at_random_ep_len=False):
    import numpy as np
    import torch
    from preflight_support import MutationGuard, state_sha256
    from pit_task import PitCommand

    if updates != 0 or init_at_random_ep_len or self.current_learning_iteration != COMPLETED:
        raise RuntimeError('Only a zero-update source34823 diagnostic is allowed')
    raw = self.env.unwrapped
    out = Path(sys.argv[sys.argv.index('--output')+1])
    source = Path(sys.argv[sys.argv.index('--resume')+1])
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    command = raw.command_manager.get_term('base_velocity')
    pit = command.pit
    if raw.num_envs != 4096 or raw.max_episode_length != 1250:
        raise RuntimeError('Must retain full 4096-environment,25-second training setup')
    before = [state_sha256(x, torch) for x in (
        self.alg.policy.state_dict(), self.alg.optimizer.state_dict(), self.alg.optimizer_stu_enc.state_dict())]
    names = ['support', 'body_clear', 'upright', 'quiet_xy', 'quiet_angular',
             'started_in_pit', 'safe', 'stop_latched', 'actual_zero_command', 'all_valid', 'success']
    n = raw.num_envs
    counts = torch.zeros((n, len(names)), device=raw.device, dtype=torch.long)
    longest = torch.zeros_like(counts)
    streaks = torch.zeros_like(counts)
    post_stop_bad = torch.zeros((n, 7), device=raw.device, dtype=torch.long)
    independent_hold = torch.zeros(n, device=raw.device, dtype=torch.long)
    maxima = torch.zeros(n, device=raw.device)
    reset_reasons = torch.zeros((n, 7), device=raw.device, dtype=torch.long)
    seen_steps = torch.zeros(n, device=raw.device, dtype=torch.long)
    command_mismatch = torch.zeros(n, device=raw.device, dtype=torch.long)
    episodes, trace = [], []
    selected = torch.cat([((pit.active) & (pit.learner == group)).nonzero().flatten()[:16]
                          for group in (0, 1)])
    refresh_original = PitCommand.refresh
    record_original = PitCommand.record_and_resample_levels
    totals_before = pit.totals.clone()
    original_step = raw.common_step_counter
    mismatch = 0

    def refreshed(c):
        nonlocal mismatch
        if c is not pit or c.last_refresh == raw.common_step_counter:
            return refresh_original(c)
        previous_stop = c.stop_on_platform.clone()
        previous_hold = independent_hold.clone()
        refresh_original(c)
        data = c.robot.data
        support = c.supported.all(-1)
        upright = -data.projected_gravity_b[:, 2] > math.cos(math.radians(15))
        xy = data.root_lin_vel_w[:, :2].norm(dim=-1)
        angular = data.root_ang_vel_w.norm(dim=-1)
        conditions = torch.stack([support, c.platform_clear, upright, xy < .15,
                                  angular < .3, c.started_in_pit, ~(c.physical | c.outside)], -1)
        valid = conditions.all(-1) & c.active
        independent_hold.copy_(torch.where(valid, previous_hold+1, 0))
        mismatch += int(((independent_hold != c.hold) & c.active).sum())
        zero = (command.command == 0).all(-1)
        values = torch.cat([conditions[:, :6], conditions[:, 6:7],
                            c.stop_on_platform[:, None], zero[:, None], valid[:, None], c.success[:, None]], -1)
        values &= c.active[:, None]
        counts.add_(values.long())
        streaks.copy_(torch.where(values, streaks+1, 0))
        longest.copy_(torch.maximum(longest, streaks))
        post_stop_bad.add_((~conditions & previous_stop[:, None] & c.active[:, None]).long())
        reset_reasons.add_((~conditions & (previous_hold > 0)[:, None] & c.active[:, None]).long())
        maxima.copy_(torch.maximum(maxima, c.hold*raw.step_dt))
        seen_steps.add_(c.active.long())
        command_mismatch.add_((previous_stop & ~zero & c.active).long())
        si = selected
        # Time series retain acceptance signals plus actual velocity and all12 leg angles/targets.
        trace.append(dict(step=raw.common_step_counter,
            values=values[si].cpu().numpy(), hold=c.hold[si].cpu().numpy(),
            command=command.command[si].cpu().numpy(), xy=xy[si].cpu().numpy(),
            angular=angular[si].cpu().numpy(), q=data.joint_pos[si].cpu().numpy(),
            force=torch.stack([s.data.force_matrix_w[:, 0, 0, 2] for s in c.ground_contacts], -1)[si].cpu().numpy(),
            episode_age=raw.episode_length_buf[si].cpu().numpy()))

    def recorded(c, ids):
        if c is not pit:
            return record_original(c, ids)
        ids = ids[raw.episode_length_buf[ids] > 0]
        c.refresh()
        for i in ids.cpu().tolist():
            episodes.append(dict(env=i, learner=int(c.learner[i]),
                level=int(raw.scene.terrain.terrain_levels[i]), speed=float(c.held_speed[i]),
                age_s=float(raw.episode_length_buf[i]*raw.step_dt), observed_steps=int(seen_steps[i]),
                conditions_steps=counts[i].cpu().tolist(), longest_steps=longest[i].cpu().tolist(),
                post_stop_bad_steps=post_stop_bad[i].cpu().tolist(), hold_reset_reasons=reset_reasons[i].cpu().tolist(),
                max_hold_s=float(maxima[i]), command_mismatch_steps=int(command_mismatch[i]),
                success=bool(c.success[i]), physical=bool(c.physical[i]), outside=bool(c.outside[i])))
        record_original(c, ids)
        for arr in (counts, longest, streaks, post_stop_bad, independent_hold, maxima,
                    reset_reasons, seen_steps, command_mismatch):
            arr[ids] = 0

    PitCommand.refresh, PitCommand.record_and_resample_levels = refreshed, recorded
    guard = MutationGuard(self)
    try:
        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        with guard, torch.inference_mode():
            for step in range(STEPS):
                action = self.alg.act(obs)
                # Diagnostic only: retain the training draws/RNG use, execute their mean on pit environments.
                # No optimizer step is permitted; do not use these altered actions for PPO updates.
                means = torch.zeros_like(action)
                ti, si = self.alg.teacher_env_idxs, self.alg.student_env_idxs
                means[ti] = self.alg.transition.action_mean[:self.alg.teacher_num_envs]
                means[si] = self.alg.transition.action_mean[self.alg.teacher_num_envs:]
                action[pit.active] = means[pit.active]
                obs, reward, done, extras = self.env.step(action.to(raw.device))
                obs, reward, done = obs.to(self.device), reward.to(self.device), done.to(self.device)
                self.alg.process_env_step(obs, reward, done, extras)
                if (step+1) % 24 == 0:
                    self.alg.storage.clear()
                if step % 150 == 0:
                    item = dict(step=step, episodes=len(episodes), hold_mismatches=mismatch,
                                max_hold_s=float(maxima.max()), ever_stop=int((counts[:, 7] > 0).sum()))
                    (out/'probe_status.json').write_text(json.dumps(item))
                    print('PIT_PROBE', json.dumps(item), flush=True)
            self.alg.storage.clear()
    finally:
        PitCommand.refresh, PitCommand.record_and_resample_levels = refresh_original, record_original
    after = [state_sha256(x, torch) for x in (
        self.alg.policy.state_dict(), self.alg.optimizer.state_dict(), self.alg.optimizer_stu_enc.state_dict())]
    if before != after or guard.calls or hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Model, optimizers or source changed during diagnostic')
    if self.current_learning_iteration != COMPLETED or list(out.rglob('*.pt')):
        raise RuntimeError('Diagnostic wrote a training artifact')
    count_delta = (pit.totals-totals_before).cpu().tolist()
    result = dict(optimizer_updates=0, completed_updates=COMPLETED, steps=STEPS,
        source_sha256=source_hash, hashes_before=before, hashes_after=after,
        hold_mismatches=mismatch, conditions=names, post_stop_bad_conditions=names[:7],
        episodes=episodes, totals_delta=count_delta, selected_envs=selected.cpu().tolist(),
        active_partial_episode_max_hold_s=float(maxima.max()),
        action_probe="pit_mean_only; training noise and randomization retained; censored15s diagnostic",
        command_class=[x.__module__+'.'+x.__name__ for x in type(command).__mro__],
        common_step_before=original_step, common_step_after=raw.common_step_counter)
    (out/'FINALIZED.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(out/'condition_traces.npz', env_ids=selected.cpu().numpy(),
                        **{k: np.stack([x[k] for x in trace]) for k in trace[0]})


early_restart.configure = configure
if __name__ == '__main__':
    from train import main
    main()
    out = Path(sys.argv[sys.argv.index('--output')+1])
    result = json.loads((out/'FINALIZED.json').read_text())
    if result['optimizer_updates'] != 0:
        raise RuntimeError('Probe did not finish')
    for name in ('SUCCESS.json', 'status.json'):
        (out/name).write_text(json.dumps(dict(state='diagnostic_complete', completed_updates=COMPLETED,
                                            optimizer_updates=0)))
