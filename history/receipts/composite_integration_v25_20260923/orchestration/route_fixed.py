"""Fixed L0 source-route cases through V25, without optimizer updates."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from contract import (BASE, HERE, NUM_ENVS, ROUTE_ENVS, SOURCE,
                      SOURCE_COMPLETED, SOURCE_SHA256, STAGE, digest, load_json,
                      training_args, verify_frozen, write_new)
from state_guard import MutationGuard, state_sha256


OUTPUT = HERE / "source_route_fixed"
SPEEDS = (0.2, 0.6, 1.0)
SEEDS = (0, 1)
MAX_STEPS = 1250


def case_ids(command, torch):
    route = command.composite.active
    groups = {}
    for learner in (0, 1):
        ids = (route & (command.learner == learner)).nonzero().flatten()[:6]
        if len(ids) != 6:
            raise RuntimeError("Fixed route requires six cases per learner")
        groups[learner] = ids
    return groups


def planned_speed(command, torch):
    groups = case_ids(command, torch)
    return [(int(groups[learner][seed * 3 + index]), learner, seed, speed)
            for learner in (0, 1) for seed in SEEDS
            for index, speed in enumerate(SPEEDS)]


def fixed_groups_ready(command) -> bool:
    route = command.composite.active
    return all(int((route & (command.learner == learner)).sum()) >= 6
               for learner in (0, 1))


def install_fixed_reset(cfg):
    """Make the migrated first episode reproducible before history is built."""
    import torch

    original_root = cfg.events.reset_base.func
    original_joints = cfg.events.reset_robot_joints.func

    def reset_root(env, ids, pose_range, velocity_range, asset_cfg):
        original_root(env, ids, pose_range, velocity_range, asset_cfg)
        command = env.command_manager.get_term("base_velocity")
        route = command.composite
        if route is None:
            return
        selected = ids[route.active[ids]]
        if not len(selected):
            return
        robot = env.scene["robot"]
        origins = env.scene.env_origins[selected]
        pose = robot.data.default_root_state[selected, :7].clone()
        pose[:, :3] += origins
        pose[:, 3:7] = 0.
        pose[:, 3] = 1.
        # Two repeatable x offsets, paired across teacher and student.
        if fixed_groups_ready(command):
            for env_id, _, seed, _ in planned_speed(command, torch):
                matched = selected == env_id
                pose[matched, 0] += .05 * seed
        points = command.pit.geometry.world_points(
            robot.data.default_joint_pos[selected], pose)
        pose[:, 2] += origins[:, 2] + .03 - points[..., 2].amin(dim=1)
        robot.write_root_pose_to_sim(pose, selected)
        robot.write_root_velocity_to_sim(torch.zeros(len(selected), 6, device=env.device), selected)

    def reset_joints(env, ids, position_range, velocity_range, asset_cfg):
        original_joints(env, ids, position_range, velocity_range, asset_cfg)
        command = env.command_manager.get_term("base_velocity")
        if command.composite is None:
            return
        selected = ids[command.composite.active[ids]]
        if len(selected):
            robot = env.scene["robot"]
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos[selected],
                torch.zeros_like(robot.data.default_joint_pos[selected]), env_ids=selected)

    cfg.events.reset_base.func = reset_root
    cfg.events.reset_robot_joints.func = reset_joints
    for group in (cfg.observations.policy, cfg.observations.single_obs,
                  cfg.observations.critic):
        group.enable_corruption = False
    cfg.events.randomize_push_robot = None
    for actuator in cfg.scene.robot.actuators.values():
        actuator.min_delay = actuator.max_delay = 2


def install_fixed_speed():
    from composite_runtime import CompositeRuntime
    import torch

    original = CompositeRuntime.resample

    def resample(self, ids):
        original(self, ids)
        if int(self.active.sum()) != ROUTE_ENVS or not fixed_groups_ready(self.command):
            return
        for env_id, _, _, speed in planned_speed(self.command, torch):
            if bool((ids == env_id).any()):
                self.held_speed[env_id] = speed
                self.command.commands[env_id] = torch.tensor(
                    (speed, 0., 0.), device=self.device)
                self.command.time_left[env_id] = 1e9

    CompositeRuntime.resample = resample


def make_evaluation(output: Path):
    def evaluate(self, updates, init_at_random_ep_len=False):
        import numpy as np
        import torch

        frozen = verify_frozen()
        if (updates != 0 or init_at_random_ep_len or self.env.num_envs != NUM_ENVS
                or self.current_learning_iteration != SOURCE_COMPLETED):
            raise RuntimeError("Fixed source diagnostics must use zero updates and 4096 V21 envs")
        raw = self.env.unwrapped
        command = raw.command_manager.get_term("base_velocity")
        route = command.composite
        plan = planned_speed(command, torch)
        selected = torch.tensor([row[0] for row in plan], device=raw.device)
        if (int(route.active.sum()) != ROUTE_ENVS
                or not bool((raw.scene.terrain.terrain_levels[selected] == 0).all())):
            raise RuntimeError("Fixed route cases are not all on new L0 terrain")
        robot = raw.scene["robot"]
        data = robot.data
        origins = raw.scene.env_origins[selected]
        initial_root = data.root_state_w[selected].clone()
        initial_root[:, :3] -= origins
        initial_joints = data.joint_pos[selected].clone()
        for a in range(6):
            b = a + 6
            torch.testing.assert_close(initial_root[a], initial_root[b], atol=2e-4, rtol=0)
            torch.testing.assert_close(initial_joints[a], initial_joints[b], atol=1e-5, rtol=0)
        speeds = torch.tensor([row[3] for row in plan], device=raw.device)
        torch.testing.assert_close(command.commands[selected, 0], speeds, atol=1e-6, rtol=0)
        if not bool((command.commands[selected, 1:] == 0).all()):
            raise RuntimeError("Fixed route cases lost straight-ahead command")
        obs = self.env.get_observations().to(self.device)
        for key in ("policy", "single_obs"):
            for a in range(6):
                torch.testing.assert_close(obs[key][selected[a]], obs[key][selected[a + 6]],
                                           atol=2e-4, rtol=0)
        self.eval_mode()
        before = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        if before != frozen["source_state_sha256"]:
            raise RuntimeError("Fixed route diagnostics did not load the exact V21 policy")
        mask = torch.zeros(NUM_ENVS, device=raw.device, dtype=torch.bool)
        mask[selected] = True
        original_reset = raw._reset_idx

        def retain_first_episode(ids):
            remaining = ids[~mask[ids]]
            if len(remaining):
                original_reset(remaining)

        raw._reset_idx = retain_first_episode
        traces = {name: [] for name in (
            "command", "root_local", "wheel_force_z", "seen_slope", "seen_stair",
            "full_body_platform", "dwell_s", "success", "physical", "outside",
            "joint_request", "joint_target", "joint_actual", "joint_torque",
            "joint_speed", "raw_action", "leg_clearance_m")}
        active = torch.ones(len(selected), device=raw.device, dtype=torch.bool)
        end_step = [-1] * len(selected)
        end_reason = [None] * len(selected)
        geometry = command.pit.geometry
        owners = geometry.owners
        leg_masks = {side: torch.tensor(
            [geometry.link_names[int(owner)].startswith(side + "_") for owner in owners],
            device=raw.device, dtype=torch.bool) for side in ("fr", "fl", "rr", "rl")}
        if any(not bool(value.any()) for value in leg_masks.values()):
            raise RuntimeError("URDF leg collision surfaces are incomplete")
        pairs = (("fr", "fl"), ("fr", "rr"), ("fr", "rl"),
                 ("fl", "rr"), ("fl", "rl"), ("rr", "rl"))
        guard = MutationGuard(self)
        with guard, torch.inference_mode():
            for step in range(MAX_STEPS):
                actions = self.alg.policy.act_inference(obs)
                teacher_ids = selected[6:]
                model = self.alg.policy
                teacher_obs = {key: value[teacher_ids] for key, value in obs.items()}
                critic = model.critic_obs_normalizer(model.get_critic_obs(teacher_obs))
                single = model.single_obs_normalizer(teacher_obs["single_obs"])
                actions[teacher_ids] = model.actor(torch.cat((model.teacher_encoder(critic), single), dim=-1))
                issued = command.commands[selected].clone()
                obs, rewards, done, extras = self.env.step(actions.to(raw.device))
                obs = obs.to(self.device)
                if not all(bool(torch.isfinite(value).all()) for value in (
                        actions[selected], data.joint_pos[selected], rewards[selected])):
                    raise FloatingPointError("Nonfinite fixed route trajectory")
                forces = torch.stack([
                    sensor.data.force_matrix_w[selected, 0, 0, 2]
                    for sensor in command.pit.ground_contacts], dim=1)
                points = geometry.world_points(data.joint_pos[selected],
                                               data.root_state_w[selected, :7])
                clearance = torch.full((len(selected),), float("inf"), device=raw.device)
                for left, right in pairs:
                    distance = torch.cdist(points[:, leg_masks[left]],
                                           points[:, leg_masks[right]]).amin(dim=(1, 2))
                    clearance = torch.minimum(clearance, distance)
                requested = raw.action_manager.get_term("joint_pos").requested_actions[selected]
                values = dict(command=issued,
                              root_local=data.root_pos_w[selected] - origins,
                              wheel_force_z=forces,
                              seen_slope=route.course.seen_slope[selected],
                              seen_stair=route.course.seen_stair[selected],
                              full_body_platform=route.course.full_body_platform[selected],
                              dwell_s=route.course.dwell_s[selected],
                              success=route.success[selected],
                              physical=route.physical[selected], outside=route.outside[selected],
                              joint_request=requested,
                              joint_target=data.joint_pos_target[selected],
                              joint_actual=data.joint_pos[selected],
                              joint_torque=data.applied_torque[selected],
                              joint_speed=data.joint_vel[selected],
                              raw_action=actions[selected], leg_clearance_m=clearance)
                for name, value in values.items():
                    traces[name].append(value.clone())
                ended = active & (route.success[selected] | route.physical[selected]
                                  | route.outside[selected]
                                  | raw.termination_manager.time_outs[selected])
                for local in ended.nonzero().flatten().tolist():
                    end_step[local] = step
                    end_reason[local] = ("success" if bool(route.success[selected[local]])
                                         else "physical" if bool(route.physical[selected[local]])
                                         else "outside" if bool(route.outside[selected[local]])
                                         else "timeout")
                active[ended] = False
                if step % 200 == 0:
                    print("ROUTE_FIXED", step, int(active.sum()), flush=True)
                if not bool(active.any()):
                    break
        raw._reset_idx = original_reset
        after = {name: state_sha256(value, torch) for name, value in (
            ("model", self.alg.policy.state_dict()),
            ("ppo_optimizer", self.alg.optimizer.state_dict()),
            ("student_optimizer", self.alg.optimizer_stu_enc.state_dict()))}
        if before != after or guard.calls or not guard.restored or digest(SOURCE) != SOURCE_SHA256:
            raise RuntimeError("Source policy or optimizer mutated during route diagnostics")
        arrays = {name: torch.stack(values).cpu().numpy() for name, values in traces.items()}
        rows = []
        for local, (env_id, learner, seed, speed) in enumerate(plan):
            until = end_step[local] + 1 if end_step[local] >= 0 else len(arrays["success"])
            distances = arrays["leg_clearance_m"][:until, local]
            minimum_at = int(np.argmin(distances))
            rows.append(dict(id=local, env_id=env_id, learner="teacher" if learner else "student",
                             seed=seed, speed_mps=speed, level=0,
                             time_feasible=speed >= route.minimum_training_speed_mps,
                             end_step=until - 1,
                             end_reason=end_reason[local] or "time_budget",
                             seen_slope=bool(arrays["seen_slope"][:until, local].any()),
                             seen_stair=bool(arrays["seen_stair"][:until, local].any()),
                             full_body_platform=bool(arrays["full_body_platform"][:until, local].any()),
                             two_second_success=bool(arrays["success"][:until, local].any()),
                             physical=bool(arrays["physical"][:until, local].any()),
                             outside=bool(arrays["outside"][:until, local].any()),
                             maximum_dwell_s=float(arrays["dwell_s"][:until, local].max()),
                             minimum_leg_clearance_m=float(distances[minimum_at]),
                             minimum_leg_clearance_step=minimum_at,
                             final_progress_m=float(arrays["root_local"][until - 1, local, 0])))
        np.savez_compressed(output / "traces.npz", **arrays,
                            initial_root=initial_root.cpu().numpy(),
                            initial_joint=initial_joints.cpu().numpy())
        write_new(output / "manifest.json", dict(
            state="complete", source=str(SOURCE), source_sha256=SOURCE_SHA256,
            source_completed_updates=SOURCE_COMPLETED, optimizer_updates=0,
            num_envs=NUM_ENVS, route_envs=ROUTE_ENVS, fixed_cases=12,
            route_level=0, speeds_mps=list(SPEEDS), seeds=list(SEEDS),
            observed_steps=len(arrays["success"]), dt_s=raw.step_dt,
            paired_initial_state_and_observations=True,
            actuator_delay_physics_steps=2, observation_corruption=False,
            no_pushing=True, state_hashes_before=before, state_hashes_after=after,
            forbidden_calls=guard.calls, mutation_guard_restored=guard.restored,
            leg_clearance_method="all sampled URDF collision points for six inter-leg pairs, every control step",
            cases=rows, fixed_cases_only=True, random_success_rate_inferred=False,
            real_robot_release=False))

    return evaluate


def main():
    import fcntl

    if sys.flags.optimize:
        raise RuntimeError("Fixed diagnostics assertions must remain enabled")
    qualified = load_json(HERE / "QUALIFIED_V25.json")
    if qualified.get("state") != "verified_not_training":
        raise RuntimeError("Independent zero-update qualification must pass first")
    verify_frozen()
    if OUTPUT.exists():
        raise FileExistsError("Fixed route diagnostics are one-shot")
    with (BASE / "full_episode_v12_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip():
            raise RuntimeError("GPU already occupied")
        sys.path.insert(0, str(STAGE / "code"))
        import early_restart

        original_configure = early_restart.configure

        def configure(cfg, profile):
            original_configure(cfg, profile)
            install_fixed_reset(cfg)
            install_fixed_speed()
            from rsl_rl.runners import OnPolicyRunnerCTS

            OnPolicyRunnerCTS.learn = make_evaluation(OUTPUT)

        early_restart.configure = configure
        prior_argv = sys.argv[:]
        sys.argv = [str(STAGE / "code/train.py"), *training_args(SOURCE, OUTPUT, 0)]
        try:
            from train import main as train_main

            train_main()
        finally:
            early_restart.configure = original_configure
            sys.argv = prior_argv
        if not (OUTPUT / "manifest.json").is_file():
            raise RuntimeError("Fixed route diagnostics did not finalize")


if __name__ == "__main__":
    main()
