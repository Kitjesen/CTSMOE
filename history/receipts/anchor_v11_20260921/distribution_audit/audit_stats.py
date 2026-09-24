"""Exact training-distribution counts plus bounded, deterministic observation samples.

All classifications use pre-step state. Terminal auto-reset observations are never
reported as the failed robot's post-step pose. Sampling does not consume policy RNG.
"""
import heapq
import json
from collections import Counter, defaultdict

import numpy as np
import torch

AGE_EDGES = np.array([.5, 1., 2., 2.5, 3., 3.3, 4.])
AGE_LABELS = ["0-.5", ".5-1", "1-2", "2-2.5", "2.5-3", "3-3.3", "3.3-4", "4+"]
DURATION_EDGES = np.array([.5, 1., 2., 3., 4.])
CLASS_NAMES = ["zero", "forward", "reverse", "lateral", "yaw", "combined"]
SIGNALS = ["actual_rms", "previous_target_rms", "previous_request_rms", "vx", "vy", "yaw_rate",
           "student_teacher_action_l2", "student_v6_action_l2", "latent_l2", "latent_cosine",
           "gate_entropy", "gate_margin", "sampled_action_rms", "nonzero_age", "segment_age"]


def stable_priority(env, episode, step):
    """Version-1 SplitMix64 priority, independent of Python hash and Torch RNG."""
    mask = (1 << 64)-1
    x = ((int(env) << 40) ^ (int(episode) << 20) ^ int(step) ^ 0xAC711D15) & mask
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30))*0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27))*0x94D049BB133111EB) & mask
    return (x ^ (x >> 31)) & mask


class Reservoir:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.heap = []
        self.serial = 0

    def accepts(self, priority):
        return len(self.heap) < self.capacity or priority < -self.heap[0][0]

    def add(self, priority, record):
        if not self.accepts(priority):
            return
        item = (-int(priority), self.serial, record)
        self.serial += 1
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
        else:
            heapq.heapreplace(self.heap, item)

    def records(self):
        return [row[2] for row in sorted(self.heap, key=lambda x: -x[0])]


def classify_commands(command):
    active = np.abs(command) > .05
    result = np.full(len(command), 5, dtype=np.int64)
    result[~active.any(1)] = 0
    only_x = active[:, 0] & ~active[:, 1] & ~active[:, 2]
    result[only_x & (command[:, 0] > 0)] = 1
    result[only_x & (command[:, 0] < 0)] = 2
    result[~active[:, 0] & active[:, 1] & ~active[:, 2]] = 3
    result[~active[:, 0] & ~active[:, 1] & active[:, 2]] = 4
    return result


class CommandClock:
    def __init__(self, count, dt):
        self.dt = dt
        self.previous_command = np.zeros((count, 3))
        self.previous_goal = np.zeros((count, 3))
        self.previous_phase = np.full(count, -999)
        self.previous_episode = np.full(count, -1)
        self.previous_nonzero = np.zeros(count, bool)
        self.nonzero_age = np.zeros(count)
        self.segment_age = np.zeros(count)

    def update(self, command, special, phase, goal, episode):
        nonzero = np.max(abs(command), axis=1) > .05
        reset = episode != self.previous_episode
        start = nonzero & (~self.previous_nonzero | reset)
        ordinary_change = np.max(abs(command-self.previous_command), axis=1) > .05
        ordinary_change |= classify_commands(command) != classify_commands(self.previous_command)
        special_change = (phase != self.previous_phase) | (np.max(abs(goal-self.previous_goal), axis=1) > 1e-6)
        changed = np.where(special, special_change, ordinary_change)
        self.nonzero_age = np.where(nonzero & ~start, self.nonzero_age+self.dt, 0.)
        self.segment_age = np.where(nonzero & ~(start | changed), self.segment_age+self.dt, 0.)
        self.previous_command = command.copy()
        self.previous_goal = goal.copy()
        self.previous_phase = phase.copy()
        self.previous_episode = episode.copy()
        self.previous_nonzero = nonzero
        return np.round(self.nonzero_age, 9), np.round(self.segment_age, 9)


class Cell:
    def __init__(self):
        self.count = 0
        self.envs = set()
        self.episodes = set()
        self.generations = defaultdict(set)
        self.total = np.zeros(len(SIGNALS))
        self.maximum = np.full(len(SIGNALS), -np.inf)
        self.ends = 0
        self.guard_ends = 0
        self.physical_ends = 0
        self.samples = Reservoir(64)

    def add(self, env, episode, values):
        self.count += 1
        self.envs.add(env)
        self.episodes.add((env, episode))
        self.generations[episode].add(env)
        self.total += values
        self.maximum = np.maximum(self.maximum, values)

    def summary(self):
        return dict(states=self.count, distinct_envs=len(self.envs), distinct_episodes=len(self.episodes),
                    episode_generations={str(k): len(v) for k, v in sorted(self.generations.items())},
                    means=dict(zip(SIGNALS, (self.total/self.count).tolist())),
                    maxima=dict(zip(SIGNALS, self.maximum.tolist())), episode_ends=self.ends,
                    guard_ends=self.guard_ends, physical_ends=self.physical_ends)


class Stats:
    def __init__(self, env, alg, reference_policy, output):
        from anchor_objective import normal_start_mask
        from startup_repair import learner_cohort_mask
        self.env, self.alg, self.reference, self.output = env, alg, reference_policy, output
        self.mask_fn = normal_start_mask
        self.step = 0
        self.episode = np.zeros(env.num_envs, dtype=np.int64)
        self.learner = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.learner[alg.student_env_idxs] = 1
        self.cohort = learner_cohort_mask(self.learner)
        torch.testing.assert_close(self.cohort, alg.anchor_cohort)
        self.clock = CommandClock(env.num_envs, env.step_dt)
        self.cells = defaultdict(Cell)
        self.critical = Cell()
        self.candidate = Cell()
        self.candidate_samples = Reservoir(4096)
        self.episode_best = {}
        self.anchor_counts = Counter()
        self.total_resets = 0
        self.total_states = 0
        self.robot = env.scene["robot"]
        self.legs = [i for i, name in enumerate(self.robot.joint_names) if not name.endswith("foot_joint")]
        self.request_ids = env.action_manager.get_term("joint_pos").joint_indices
        self.pending = None
        self.term_snapshot = None
        manager = env.termination_manager
        self._original_termination_compute = manager.compute
        self.termination_timeouts = {name: bool(manager.get_term_cfg(name).time_out)
                                     for name in manager.active_terms}

        def capture_termination(*args, **kwargs):
            result = self._original_termination_compute(*args, **kwargs)
            # Capture causes before the environment auto-resets terminal rows.
            # The original result is returned unmodified to preserve rollout semantics.
            self.term_snapshot = {name: manager.get_term(name).detach().cpu().numpy().copy()
                                  for name in manager.active_terms}
            return result

        manager.compute = capture_termination

    def close(self):
        self.env.termination_manager.compute = self._original_termination_compute

    @torch.no_grad()
    def before(self, obs, actions):
        from anchor_objective import deterministic_mean
        env, model = self.env, self.alg.policy
        rng_before = torch.cuda.get_rng_state().clone()
        command = env.command_manager.get_term("base_velocity")
        flat = command.terrain_idxs == command.terrain_type2idx["flat"]
        ids = flat.nonzero().flatten()
        age = env.episode_length_buf*env.step_dt
        special = flat & self.cohort
        goal = command.start_goal.detach().cpu().numpy()
        cmd = command.command.detach().cpu().numpy().copy()
        phase = command.start_phase.detach().cpu().numpy()
        nz_age, seg_age = self.clock.update(cmd, special.cpu().numpy(), phase, goal, self.episode)
        torch.testing.assert_close(obs["single_obs"][:, 6:9], command.command, rtol=0, atol=1e-6)
        part = obs[ids]
        single = model.single_obs_normalizer(part["single_obs"])
        zs, gate = model.student_moe_encoder(model.actor_obs_normalizer(model.get_actor_obs(part)))
        zt = model.teacher_encoder(model.critic_obs_normalizer(model.get_critic_obs(part)))
        student = model.actor(torch.cat((zs, single), -1))
        teacher = model.actor(torch.cat((zt, single), -1))
        reference = deterministic_mean(self.reference, part, is_teacher=False)
        data = self.robot.data
        q = data.joint_pos[ids][:, self.legs]-data.default_joint_pos[ids][:, self.legs]
        target = data.joint_pos_target[ids][:, self.legs]-data.default_joint_pos[ids][:, self.legs]
        request = env.action_manager.get_term("joint_pos").requested_actions[ids]-data.default_joint_pos[ids][:, self.request_ids]
        sorted_gate = gate.sort(-1).values
        core = torch.stack((q.square().mean(-1).sqrt(), target.square().mean(-1).sqrt(),
            request.square().mean(-1).sqrt(), data.root_lin_vel_b[ids, 0], data.root_lin_vel_b[ids, 1],
            data.root_ang_vel_b[ids, 2], (student-teacher).norm(dim=-1), (student-reference).norm(dim=-1),
            (zs-zt).norm(dim=-1), torch.nn.functional.cosine_similarity(zs, zt, dim=-1),
            -(gate*gate.clamp_min(1e-12).log()).sum(-1), sorted_gate[:, -1]-sorted_gate[:, -2],
            actions[ids].square().mean(-1).sqrt()), -1).cpu().numpy()
        env_ids = ids.cpu().numpy()
        age_np = age[ids].cpu().numpy()
        age_bins = np.searchsorted(AGE_EDGES, age_np, side="right")
        classes = classify_commands(cmd[env_ids])
        magnitudes = np.where(np.max(abs(cmd[env_ids, :2]), axis=1) > .05,
                              np.linalg.norm(cmd[env_ids, :2], axis=1), abs(cmd[env_ids, 2]))
        magnitude_bins = np.searchsorted([.45, .75, 1.2], magnitudes, side="right")
        learner_np = self.learner[ids].cpu().numpy()
        special_np = special[ids].cpu().numpy()
        holds = command.start_hold[ids].cpu().numpy()
        periods = command.start_period[ids].cpu().numpy()
        ramps = command.start_ramp[ids].cpu().numpy()
        time_left = command.time_left[ids].cpu().numpy()
        phases = np.where(~special_np, -1, np.where(age_np < holds, 0,
                           np.where(phase[env_ids] % 2 == 0, 1, 2)))
        original_mask = (self.mask_fn(flat, age, command.command) & self.cohort)[ids].cpu().numpy()
        records = {key: value.detach().cpu().numpy() for key, value in part.items()}
        telemetry = {"joint_pos": data.joint_pos[ids].cpu().numpy(),
                     "previous_target": data.joint_pos_target[ids].cpu().numpy(),
                     "previous_request": env.action_manager.get_term("joint_pos").requested_actions[ids].cpu().numpy(),
                     "sampled_action": actions[ids].cpu().numpy(), "student_action": student.cpu().numpy(),
                     "teacher_action": teacher.cpu().numpy(), "reference_action": reference.cpu().numpy(),
                     "student_latent": zs.cpu().numpy(), "teacher_latent": zt.cpu().numpy(), "gate": gate.cpu().numpy()}
        if not np.isfinite(core).all():
            raise FloatingPointError("Non-finite pre-step audit telemetry")
        pending = []
        for row, eid in enumerate(env_ids.tolist()):
            ep = int(self.episode[eid])
            key = (int(learner_np[row]), int(special_np[row]), int(age_bins[row]), int(phases[row]),
                   int(classes[row]), int(magnitude_bins[row]),
                   int(np.searchsorted(DURATION_EDGES, nz_age[eid], side="right")),
                   int(np.searchsorted(DURATION_EDGES, seg_age[eid], side="right")))
            values = np.concatenate((core[row], [nz_age[eid], seg_age[eid]]))
            cell = self.cells[key]
            cell.add(eid, ep, values)
            self.anchor_counts[(int(learner_np[row]), int(original_mask[row]))] += 1
            eligible = (learner_np[row] == 1 and special_np[row] and phases[row] == 1
                        and 1. <= age_np[row] < 3. and .45 <= cmd[eid, 0] <= .75
                        and abs(cmd[eid, 1]) <= .05 and abs(cmd[eid, 2]) <= .05)
            critical = eligible and 2.5 <= age_np[row] < 3.
            if eligible:
                self.candidate.add(eid, ep, values)
            if critical:
                self.critical.add(eid, ep, values)
            priority = stable_priority(eid, ep, self.step)
            episode_key = (eid, ep)
            episode_accept = eligible and (episode_key not in self.episode_best or priority < self.episode_best[episode_key][0])
            record = None
            if cell.samples.accepts(priority) or (eligible and self.candidate_samples.accepts(priority)) or episode_accept:
                record = dict(env_id=eid, episode=ep, step=self.step, cell=key, age=float(age_np[row]),
                    command=cmd[eid].copy(), start_goal=goal[eid].copy(), start_phase=int(phase[eid]),
                    start_hold=float(holds[row]), start_period=float(periods[row]),
                    start_ramp=bool(ramps[row]), time_left=float(time_left[row]),
                    anchor_mask=bool(original_mask[row]), signals=values.copy(),
                    **{f"obs_{k}": v[row].copy() for k, v in records.items()},
                    **{k: v[row].copy() for k, v in telemetry.items()})
                cell.samples.add(priority, record)
                if eligible:
                    self.candidate_samples.add(priority, record)
                    if episode_accept:
                        self.episode_best[episode_key] = (priority, record)
            pending.append((eid, cell, record, eligible, critical))
        self.total_states += len(env_ids)
        self.pending = pending
        self.term_snapshot = None
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng_before)

    def after(self, obs, rewards, dones, extras):
        done = dones.cpu().numpy().astype(bool)
        violation = self.env.actuator_guard.violation.cpu().numpy()
        if self.term_snapshot is None:
            raise RuntimeError("Termination causes were not captured before auto-reset")
        physical = np.zeros_like(done)
        for name, values in self.term_snapshot.items():
            if not self.termination_timeouts[name]:
                physical |= values
        data = self.robot.data
        post_q = data.joint_pos.cpu().numpy()
        post_target = data.joint_pos_target.cpu().numpy()
        post_request = self.env.action_manager.get_term("joint_pos").requested_actions.cpu().numpy()
        for eid, cell, record, eligible, critical in self.pending:
            for dest in [cell, *([self.candidate] if eligible else []), *([self.critical] if critical else [])]:
                dest.ends += int(done[eid])
                dest.guard_ends += int(done[eid] and violation[eid])
                dest.physical_ends += int(done[eid] and physical[eid])
            if record is not None:
                record["done"] = bool(done[eid])
                record["guard_violation"] = bool(violation[eid])
                record["termination_terms"] = [name for name, values in self.term_snapshot.items() if values[eid]]
                record["post_state_valid"] = not bool(done[eid])
                if not done[eid]:
                    record["post_joint_pos"] = post_q[eid].copy()
                    record["post_target"] = post_target[eid].copy()
                    record["post_request"] = post_request[eid].copy()
        self.episode += done
        self.total_resets += int(done.sum())
        self.step += 1
        self.pending = None

    def finish(self):
        if self.pending is not None:
            raise RuntimeError("Unpaired before/after audit callback")
        self.close()
        self.output.mkdir(parents=True, exist_ok=True)
        rows = [dict(cell=list(key), **cell.summary()) for key, cell in sorted(self.cells.items())]
        critical = self.critical.summary() if self.critical.count else None
        coverage = bool(critical and critical["states"] >= 256 and critical["distinct_envs"] >= 32
                        and len(critical["episode_generations"]) >= 2)
        summary = dict(steps=self.step, num_envs=self.env.num_envs, total_flat_states=self.total_states,
            total_resets_all_terrains=self.total_resets, groups=rows,
            key_dimensions=["learner_0teacher_1student", "special", "age_bin", "phase_-1ordinary_0hold_1move_2stop",
                            "command_class", "magnitude_bin", "nonzero_duration_bin", "segment_duration_bin"],
            age_bins=AGE_LABELS, command_classes=CLASS_NAMES, magnitude_edges=[.45,.75,1.2],
            magnitude_units="XY command norm in m/s when translating, otherwise yaw rad/s",
            duration_edges=DURATION_EDGES.tolist(),
            signal_timing="pre-step actual pose, prior held request/target, current sampled action and counterfactuals",
            counts_are_correlated_states_not_independent_trials=True, optimizer_steps=0,
            critical_coverage=critical, coverage_gate=coverage,
            coverage_rule="student special moving, age [2.5,3), vx [.45,.75], |vy|/|yaw|<=.05; >=256 states, >=32 envs, >=2 episode ordinals",
            candidate=self.candidate.summary() if self.candidate.count else None,
            original_anchor_counts=[dict(learner=k[0], anchor=bool(k[1]), states=v) for k,v in sorted(self.anchor_counts.items())])
        (self.output/"distribution.json").write_text(json.dumps(summary, indent=2))
        # Tensor records preserve bounded observation samples without JSON float expansion.
        # No model/optimizer or original checkpoint is written by this diagnostic.
        cell_records = [record for cell in self.cells.values() for record in cell.samples.records()]
        balanced = Reservoir(64)
        for (eid, ep), (_, record) in self.episode_best.items():
            balanced.add(stable_priority(eid, ep, 0), record)
        samples = dict(cells=cell_records, candidate_state_weighted=self.candidate_samples.records(),
                       candidate_episode_balanced=balanced.records(),
                       default_q=self.robot.data.default_joint_pos[0].cpu().numpy(),
                       joint_names=self.robot.joint_names, priority_version="splitmix64-v1")
        path = self.output/"samples.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save(samples, path)
        print("DISTRIBUTION_AUDIT", json.dumps(dict(steps=self.step, coverage=coverage,
            critical=critical, stored_cell_samples=len(cell_records),
            candidate_samples=len(samples["candidate_state_weighted"]))), flush=True)
        return summary
