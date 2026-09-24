"""Official Go2 CTS learning loop with isolated Thunder configuration and receipts."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from training_profiles import configure_base_height_penalty, rebalance_terrain_assignments, summarize_terrain


def apply_command_curriculum_start(command_cfg, start):
    """Start at the existing official 20k stage; preserve subsequent stages."""
    if start == "original":
        return
    if start != "intermediate":
        raise ValueError(f"Unknown command curriculum start: {start}")
    stage = next(item for item in command_cfg.command_range_curriculum if item["iter"] == 20000)
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"):
        setattr(command_cfg.ranges, axis, tuple(stage[axis]))
    command_cfg.command_range_curriculum = [
        item for item in command_cfg.command_range_curriculum if item["iter"] > stage["iter"]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fast-delay", action="store_true")
    parser.add_argument("--mixed-pit", action="store_true", help="Resume original model with nine terrain training")
    parser.add_argument("--expand-flat-speed", action="store_true",
                        help="Flat vx +/-1.5 to +/-2 m/s over 2000 updates; inherited on resume")
    parser.add_argument("--uphill-speed-limit", type=float, choices=(1.2, 1.5),
                        help="Advance uphill vx only; inherit on resume and retain later official stages")
    parser.add_argument("--pit-posture-profile", choices=("off", "phase_stance_v1"),
                        help="Normal stance before and after climbing; inherits on resume")
    parser.add_argument("--surface-posture-profile", choices=("off", "level_surface_v1"),
                        help="Normal stance on locally level old terrain; inherits on resume")
    parser.add_argument("--stance-strength-profile", choices=("off", "strong_stand_v1"),
                        help="Stronger height and zero-command stance objectives; inherits on resume")
    parser.add_argument("--allocation-profile", choices=("fixed", "balanced_adaptive_v1"),
                        help="Default inherits checkpoint; allocation changes only at episode resets")
    parser.add_argument("--pit-course-profile", choices=("global_v1", "individual_speed_v1"),
                        help="Default inherits checkpoint; individual progression within each speed bin")
    parser.add_argument("--pit-training-profile", choices=("strict_v1", "climb_first_v2"),
                        help="Default inherits checkpoint; climb-first separates climbing from parking quality")
    parser.add_argument("--actuator-profile", choices=("original", "rs0402_guard_v1", "rs0402_peak_v2", "rs0402_slew12_v3", "rs0402_transient_v4"),
                        help="Default inherits checkpoint; guarded RS04/RS02 experiment budgets")
    parser.add_argument("--slope-profile", choices=("original", "slope45_v1"),
                        help="Default inherits checkpoint; explicit angle curriculum through 45 degrees")
    parser.add_argument("--command-curriculum-start", choices=("original", "intermediate"),
                        help="Start at original or official 20k command ranges; default inherits resumed checkpoint")
    parser.add_argument("--base-height-penalty", choices=("original", "off"),
                        help="Default inherits resumed checkpoint")
    parser.add_argument("--terrain-training-profile", choices=("original", "stairs80_v1"),
                        help="Default inherits resumed checkpoint; stairs80 uses 60%% upstairs and 20%% downstairs")
    parser.add_argument("--stair-command-profile", choices=("original", "straight_speed_v1"),
                        help="Default inherits resumed checkpoint; straight_speed_v1 holds sampled positive speed on stairs")
    parser.add_argument("--wave-reset-profile", choices=("original", "wave_surface_v1"),
                        help="Default inherits checkpoint; wave_surface_v1 places wave collision surfaces near terrain")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output directory must be new")
    args.output.mkdir(parents=True)
    root = args.root.resolve()
    sys.path[:0] = [str(root / "upstream/source/rsl_rl"), str(root / "upstream/source/robot_lab"), str(root / "thunder/source/doso_train")]
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        import torch
        from tensordict import TensorDict
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from robot_lab.tasks.go2.env.go2_env import Go2Env
        from robot_lab.tasks.go2.env_cfg import Go2EnvCfg
        from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
        from robot_lab.tasks.go2.mdp.utils import _get_terrain_column_range
        from rsl_rl.runners import OnPolicyRunnerCTS
        from config import make_config

        torch.set_num_threads(4)
        saved = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
        from flat_speed import snapshot as flat_speed_snapshot
        from uphill_speed import snapshot as uphill_speed_snapshot
        uphill_limit = args.uphill_speed_limit or (saved.get("uphill_speed_limit") if saved else None)
        if uphill_limit is not None and not args.mixed_pit:
            raise ValueError("Uphill speed expansion requires the mixed command task")
        flat_course = saved.get("flat_speed_course") if saved else None
        if args.expand_flat_speed and flat_course is None:
            flat_course = dict(profile="flat_1p5_to_2_v1", start_step=saved["common_step_counter"] if saved else 0)
        if flat_course is not None and not args.mixed_pit:
            raise ValueError("Flat speed expansion requires the mixed command task")
        from policy_contract import checkpoint_profile, export_contract
        source_profile = checkpoint_profile(saved) if saved else "original"
        actuator_profile = args.actuator_profile or source_profile
        command_start = args.command_curriculum_start or (
            saved.get("command_curriculum_start", "original") if saved else "original"
        )
        old_terrain_profile = saved.get("terrain_training_profile", "original") if saved else "original"
        terrain_profile = args.terrain_training_profile or old_terrain_profile
        height_penalty = args.base_height_penalty or (
            saved.get("base_height_penalty", "original") if saved else "original"
        )
        stair_profile = args.stair_command_profile or (
            saved.get("stair_command_profile", "original") if saved else "original"
        )
        wave_profile = args.wave_reset_profile or (saved.get("wave_reset_profile", "original") if saved else "original")
        old_slope_profile = saved.get("slope_profile", "original") if saved else "original"
        old_pit_profile = saved.get("pit_training_profile", "strict_v1") if saved else "strict_v1"
        posture_profile = args.pit_posture_profile or (saved.get("pit_posture_profile", "off") if saved else "off")
        surface_profile = args.surface_posture_profile or (saved.get("surface_posture_profile", "off") if saved else "off")
        stance_strength = args.stance_strength_profile or (saved.get("stance_strength_profile", "off") if saved else "off")
        if posture_profile != "off" and not args.mixed_pit:
            raise ValueError("Pit posture requires mixed terrain")
        pit_profile = args.pit_training_profile or old_pit_profile
        old_pit_course_profile = saved.get("pit_course_profile", "global_v1") if saved else "global_v1"
        pit_course_profile = args.pit_course_profile or old_pit_course_profile
        if old_pit_course_profile != "global_v1" and pit_course_profile != old_pit_course_profile:
            raise ValueError("Reverse individual pit curriculum migration requires an explicit design")
        if pit_course_profile != "global_v1" and (not args.mixed_pit or pit_profile != "climb_first_v2"):
            raise ValueError("Individual course requires mixed climb-first training")
        old_allocation_profile = saved.get("allocation_profile", "fixed") if saved else "fixed"
        allocation_profile = args.allocation_profile or old_allocation_profile
        allocation = None
        if allocation_profile != "fixed" and not args.mixed_pit:
            raise ValueError("Adaptive allocation requires mixed terrain")
        if pit_profile != "strict_v1" and not args.mixed_pit:
            raise ValueError("Climb-first profile requires the mixed task")
        slope_profile = args.slope_profile or old_slope_profile
        if slope_profile != "original" and not args.mixed_pit:
            raise ValueError("Slope45 requires the mixed terrain task")
        cfg = make_config(args.num_envs, root / "cache/usd", fast_delay=args.fast_delay,
                          actuator_profile=actuator_profile)
        apply_command_curriculum_start(cfg.commands.base_velocity, command_start)
        configure_base_height_penalty(cfg, height_penalty)
        if stair_profile != "original":
            from stair_speed import configure_stair_speed
            configure_stair_speed(cfg, stair_profile)
        from wave_reset import configure_wave_reset, reset_snapshot
        configure_wave_reset(cfg, wave_profile)
        if args.mixed_pit:
            if saved is None or stair_profile != "straight_speed_v1" or wave_profile != "wave_surface_v1":
                raise ValueError("Mixed task requires the mature stairs/wave checkpoint")
            from mixed_task import configure_mixed
            configure_mixed(cfg, slope_profile=slope_profile, pit_training_profile=pit_profile)
            cfg.pit_course_profile = pit_course_profile
        from pit_posture import configure as configure_pit_posture, snapshot as posture_snapshot
        configure_pit_posture(cfg, posture_profile)
        from surface_posture import configure as configure_surface_posture, snapshot as surface_snapshot
        configure_surface_posture(cfg, surface_profile)
        from stance_strength import configure as configure_stance_strength, snapshot as stance_snapshot
        configure_stance_strength(cfg, stance_strength)
        cfg.seed = 42
        cfg.flat_speed_course = flat_course
        cfg.uphill_speed_limit = uphill_limit
        cfg.log_dir = str(args.output)
        agent = MoECTSRunnerCfg()
        agent.seed = 42
        agent.max_iterations = args.updates
        agent.logger = "tensorboard"
        agent.device = "cuda:0"
        agent_dict = agent.to_dict()
        agent_dict["robogauge"] = {"enabled": False}
        dump_yaml(str(args.output / "upstream_env.yaml"), Go2EnvCfg())
        dump_yaml(str(args.output / "env.yaml"), cfg)
        dump_yaml(str(args.output / "agent.yaml"), agent_dict)
        raw = Go2Env(cfg=cfg)
        if wave_profile != "original":
            reset_terms = raw.event_manager.active_terms["reset"]
            assert reset_terms.index("wave_surface") > max(reset_terms.index(name) for name in ("reset_robot_joints", "reset_base"))
        guard = None
        if actuator_profile != "original":
            from actuator_guard import ActuatorGuard
            guard = ActuatorGuard(raw)
        terrain_cfg = cfg.scene.terrain.terrain_generator
        column_names = [None] * terrain_cfg.num_cols
        for name in terrain_cfg.sub_terrains:
            first, last = _get_terrain_column_range(terrain_cfg, name, raw.device)
            for col in range(first, last):
                assert column_names[col] is None, (name, col)
                column_names[col] = name
        assert all(column_names)
        if saved and "terrain_column_names" in saved:
            expected_columns = column_names[:20] if args.mixed_pit and not saved.get("mixed_profile") else column_names
            assert saved["terrain_column_names"] == expected_columns, "Original terrain columns changed"
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        obs = env.get_observations()
        dimensions = {key: list(value.shape) for key, value in obs.items()}
        assert obs["policy"].shape == (args.num_envs, 530), dimensions
        assert obs["single_obs"].shape == (args.num_envs, 53), dimensions
        assert obs["critic"].shape == (args.num_envs, 291), dimensions
        assert env.num_actions == 16
        assert raw.scene.terrain.terrain_origins.shape == ((22, 40, 3) if args.mixed_pit else (10, 20, 3))
        assert "illegal_contact" in raw.termination_manager.active_terms
        assert hasattr(raw.action_manager, "prev_prev_action")
        print("CONTRACT_OK", dimensions, flush=True)
        (args.output / "contract.json").write_text(json.dumps(dimensions, indent=2))

        class StudentExport(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.encoder = copy.deepcopy(policy.student_moe_encoder).cpu().eval()
                self.actor = copy.deepcopy(policy.actor).cpu().eval()

            def forward(self, history, single_obs):
                latent, _ = self.encoder(history)
                return self.actor(torch.cat((latent, single_obs), dim=-1))

        class ReceiptRunner(OnPolicyRunnerCTS):
            def save(self, path, it, last_model, infos=None):
                # Keep upstream checkpoint format, add explicit completed count and curriculum state.
                p = Path(path)
                super().save(str(p) + ".tmp", it, last_model, infos)
                saved = torch.load(str(p) + ".tmp", weights_only=False)
                saved.update(completed_updates=it + 1,
                             pit_posture_profile=posture_profile,
                             surface_posture_profile=surface_profile,
                             stance_strength_profile=stance_strength,
                             flat_speed_course=flat_course,
                             uphill_speed_limit=uphill_limit,
                             common_step_counter=raw.common_step_counter,
                             terrain_levels=raw.scene.terrain.terrain_levels,
                             terrain_types=raw.scene.terrain.terrain_types,
                             learning_rate=self.alg.learning_rate,
                             command_curriculum_start=command_start,
                             terrain_training_profile=terrain_profile,
                             terrain_column_names=column_names,
                             base_height_penalty=height_penalty,
                             stair_command_profile=stair_profile,
                             wave_reset_profile=wave_profile,
                             slope_profile=slope_profile,
                             pit_training_profile=pit_profile,
                             pit_course_profile=pit_course_profile,
                             allocation_profile=allocation_profile,
                             actuator_profile=actuator_profile,
                             actuator_contract=export_contract(actuator_profile))
                if stair_profile != "original":
                    saved["stair_episode_totals"] = command.episode_totals
                if args.mixed_pit:
                    saved.update(mixed_profile="mixed9_v1", pit_course=command.pit.course.state_dict(),
                                 pit_totals=command.pit.totals, pit_climb_totals=command.pit.climb_totals)
                if allocation is not None:
                    saved["terrain_allocation"] = allocation.state_dict()
                if args.mixed_pit and command.pit.individual is not None:
                    saved["pit_individual"] = command.pit.individual.state_dict()
                torch.save(saved, str(p) + ".tmp")
                os.replace(str(p) + ".tmp", p)
                export = StudentExport(self.alg.policy)
                # Saving must not resample noise or touch observation histories.
                batch = TensorDict(raw.obs_buf, batch_size=[args.num_envs])[:8]
                history, single = batch["policy"].cpu(), batch["single_obs"].cpu()
                with torch.no_grad():
                    expected = self.alg.policy.act_inference(batch).cpu()
                    torch.testing.assert_close(export(history, single), expected, atol=3e-5, rtol=3e-5)
                    traced = torch.jit.trace(export, (history, single))
                    torch.testing.assert_close(traced(history, single), expected, atol=3e-5, rtol=3e-5)
                    traced.save(str(p.with_name(f"student_{it + 1}.pt")), _extra_files={
                        "actuator_contract.json": json.dumps(export_contract(actuator_profile)),
                        "training_context.json": json.dumps({"wave_reset_profile": wave_profile,
                            "slope_profile": slope_profile,
                            "pit_training_profile": pit_profile,
                            "pit_course_profile": pit_course_profile,
                            "mixed_profile": "mixed9_v1" if args.mixed_pit else None})})
                print("SAVE_EXPORT_OK", it + 1, flush=True)

        runner = ReceiptRunner(env, agent_dict, log_dir=str(args.output), device="cuda:0")
        resume_checks = {}
        if args.resume:
            runner.load(str(args.resume), map_location="cpu")
            for name, actual, expected in (
                ("model", runner.alg.policy.state_dict(), saved["model_state_dict"]),
                ("ppo_optimizer", runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"]),
                ("student_optimizer", runner.alg.optimizer_stu_enc.state_dict(), saved["optimizer_stu_enc_state_dict"]),
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_device=False)
                resume_checks[name] = True
            runner.current_learning_iteration = saved["completed_updates"]
            runner.alg.learning_rate = saved["learning_rate"]
            raw.common_step_counter = saved["common_step_counter"]
        terrain = raw.scene.terrain
        command = raw.command_manager.get_term("base_velocity")
        target_levels = saved["terrain_levels"].tolist() if saved else terrain.terrain_levels.cpu().tolist()
        target_columns = saved["terrain_types"].tolist() if saved else terrain.terrain_types.cpu().tolist()
        migration = None
        learner_groups = {"teacher": runner.alg.teacher_env_idxs.cpu().tolist(),
                          "student": runner.alg.student_env_idxs.cpu().tolist()}
        if stair_profile != "original":
            command.set_student_indices(runner.alg.student_env_idxs)
        if args.mixed_pit and not saved.get("mixed_profile"):
            from mixed_logic import mixed_assignments
            target_levels, target_columns, initial_allocation = mixed_assignments(
                target_levels, target_columns, saved["terrain_column_names"], learner_groups)
            migration = {"profile": "mixed9_v1", "group_counts": initial_allocation,
                         "original_levels": "retained environments keep their exact saved levels",
                         "pit_levels": "new environments begin at 25cm"}
            (args.output/"terrain_migration.json").write_text(json.dumps(migration, indent=2))
        elif args.mixed_pit:
            from pit_logic import PitCourse
            if saved["mixed_profile"] != "mixed9_v1":
                raise ValueError("Unknown mixed task checkpoint")
            command.pit.course = PitCourse.restore(saved["pit_course"])
            command.pit.totals.copy_(saved["pit_totals"].to(raw.device))
            if old_pit_profile != pit_profile:
                if old_pit_profile != "strict_v1" or pit_profile != "climb_first_v2":
                    raise ValueError("Unsupported pit course migration")
                old_course = command.pit.course
                command.pit.course = PitCourse(frontier=old_course.frontier, max_level=old_course.max_level,
                                              min_samples=old_course.min_samples)
                (args.output/"pit_course_migration.json").write_text(json.dumps(dict(
                    from_profile=old_pit_profile, to_profile=pit_profile, retained_frontier=old_course.frontier,
                    reason="Reset gate batches because their success definition changed; retain physical heights"), indent=2))
            elif pit_profile == "climb_first_v2":
                command.pit.climb_totals.copy_(saved["pit_climb_totals"].to(raw.device))
                torch.testing.assert_close(command.pit.climb_totals.cpu(), saved["pit_climb_totals"].cpu(), rtol=0, atol=0)
                resume_checks["pit_climb_totals"] = True
        elif old_terrain_profile != terrain_profile:
            before_summary = summarize_terrain(target_levels, target_columns, column_names)
            target_levels, target_columns, migration = rebalance_terrain_assignments(
                target_levels, target_columns, column_names, learner_groups, terrain_profile,
            )
            migration.update(from_profile=old_terrain_profile, before=before_summary,
                             after=summarize_terrain(target_levels, target_columns, column_names))
            (args.output / "terrain_migration.json").write_text(json.dumps(migration, indent=2))
        slope_migration = None
        if allocation_profile != old_allocation_profile:
            if allocation_profile != "balanced_adaptive_v1":
                raise ValueError("Explicit reverse allocation migration is not supported")
            from terrain_allocation import BASE, reassign
            target_levels, target_columns, migration = reassign(
                target_levels, target_columns, column_names, learner_groups, BASE)
            (args.output/"terrain_migration.json").write_text(json.dumps(migration, indent=2))
        if old_slope_profile != slope_profile:
            from slope_curriculum import migrate_slope_levels
            target_levels, slope_migration = migrate_slope_levels(
                target_levels, target_columns, column_names, old_slope_profile, slope_profile)
            (args.output/"slope_migration.json").write_text(json.dumps(slope_migration, indent=2))
        if args.resume or migration:
            terrain.terrain_levels[:] = torch.tensor(target_levels, device=raw.device)
            terrain.terrain_types[:] = torch.tensor(target_columns, device=raw.device)
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            raw.scene.env_origins[:] = terrain.env_origins
            # Commands cache terrain membership: refresh before the first resample.
            command._init_terrain_infos()
            command._update_env_command_ranges()
            if args.mixed_pit:
                command.refresh_membership()
            if pit_course_profile != "global_v1":
                from pit_individual import IndividualPitCourse
                command.pit.individual = IndividualPitCourse(terrain.terrain_levels, command.pit.course.max_level)
                if old_pit_course_profile == pit_course_profile:
                    command.pit.individual.load(saved["pit_individual"])
                    for name, value in saved["pit_individual"].items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(getattr(command.pit.individual, name).cpu(), value.cpu(), rtol=0, atol=0)
                    resume_checks["pit_individual"] = True
                else:
                    (args.output/"pit_individual_migration.json").write_text(json.dumps(dict(
                        profile=pit_course_profile, initial_levels="exact saved height per environment copied into four speed banks",
                        old_global_frontier=command.pit.course.frontier, success_streak=2,
                        failure_streak=2, review_probability=.2, max_height_m=1.2), indent=2))
            expected_terrain_idxs = torch.tensor(
                [command.terrain_type2idx[column_names[col]] for col in target_columns], device=raw.device,
            )
            torch.testing.assert_close(command.terrain_idxs, expected_terrain_idxs, rtol=0, atol=0)
            compute = raw.curriculum_manager.compute
            raw.curriculum_manager.compute = lambda env_ids: None
            try:
                raw.reset()
            finally:
                raw.curriculum_manager.compute = compute
            # Restore mature reward courses immediately, without advancing terrain.
            for name in ("base_linear_velocity", "base_height_l2"):
                term = getattr(cfg.curriculum, name)
                if term is not None:
                    term.func(raw, None, **term.params)
            if args.mixed_pit:
                command.pit.last_refresh = -1
            assert terrain.terrain_levels.cpu().tolist() == target_levels
            assert terrain.terrain_types.cpu().tolist() == target_columns
            resume_checks["terrain_assignment"] = True
            resume_checks["command_terrain_membership"] = True
            resume_checks["terrain_restored_exactly"] = migration is None and slope_migration is None
        if stair_profile != "original" and saved and "stair_episode_totals" in saved:
            command.episode_totals.copy_(saved["stair_episode_totals"].to(raw.device))
            torch.testing.assert_close(command.episode_totals.cpu(), saved["stair_episode_totals"].cpu(), rtol=0, atol=0)
            resume_checks["stair_episode_totals"] = True
        if args.resume:
            print("RESUME_OK", runner.current_learning_iteration, flush=True)
        if height_penalty == "off":
            assert cfg.curriculum.base_height_l2 is None
            assert raw.reward_manager.get_term_cfg("base_height_l2").weight == 0
        mixed_checks = None
        if args.mixed_pit:
            from mixed_task import validate_mixed_runtime
            mixed_checks = validate_mixed_runtime(raw, sum(column_names[col] == "pit" for col in target_columns))
        if allocation_profile != "fixed":
            from terrain_allocation_runtime import AdaptiveAllocation
            allocation = AdaptiveAllocation(raw, column_names, learner_groups,
                runner.current_learning_iteration,
                saved.get("terrain_allocation") if old_allocation_profile == allocation_profile else None)
            raw.terrain_allocation = allocation
            if old_allocation_profile == allocation_profile:
                torch.testing.assert_close(allocation.stats.cpu(), saved["terrain_allocation"]["stats"].cpu(), rtol=0, atol=0)
                assert allocation.course.state_dict() == saved["terrain_allocation"]["course"]
                assert allocation.pending == saved["terrain_allocation"]["pending"]
                resume_checks["terrain_allocation"] = True
        run_receipt = dict(
            resume=str(args.resume) if args.resume else None,
            resume_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
            starting_completed_updates=saved["completed_updates"] if saved else 0,
            additional_updates=args.updates,
            target_completed_updates=(saved["completed_updates"] if saved else 0) + args.updates,
            command_curriculum_start=command_start, command_ranges=command.command_ranges,
            flat_speed=flat_speed_snapshot(command), flat_speed_course=flat_course,
            uphill_speed=uphill_speed_snapshot(command), uphill_speed_limit=uphill_limit,
            resume_checks=resume_checks, episode_length_s=cfg.episode_length_s,
            terrain_training_profile=terrain_profile, base_height_penalty=height_penalty,
            stair_command_profile=stair_profile,
            wave_reset_profile=wave_profile, wave_reset=reset_snapshot(raw),
            actuator_profile=actuator_profile,
            actuator_guard=guard.snapshot() if guard is not None else None,
            leg_target_max_delta_rad=float(raw.action_manager.get_term("joint_pos")._max_delta) if guard else None,
            initial_episode_lengths_randomized=stair_profile == "original",
            stair_speed=command.snapshot() if stair_profile != "original" else None,
            terrain_column_names=column_names,
            terrain_by_type=summarize_terrain(target_levels, target_columns, column_names, slope_profile),
            terrain_by_learner={group: summarize_terrain([target_levels[i] for i in ids],
                                                        [target_columns[i] for i in ids], column_names, slope_profile)
                                for group, ids in learner_groups.items()},
            terrain_migration="terrain_migration.json" if migration else None,
            mixed_profile="mixed9_v1" if args.mixed_pit else None,
            mixed_checks=mixed_checks,
            slope_profile=slope_profile,
            pit_training_profile=pit_profile,
            pit_posture_profile=posture_profile, pit_posture=posture_snapshot(raw),
            surface_posture_profile=surface_profile, surface_posture=surface_snapshot(raw),
            stance_strength_profile=stance_strength, stance_strength=stance_snapshot(raw),
            pit_course_profile=pit_course_profile,
            allocation_profile=allocation_profile,
            terrain_allocation=allocation.snapshot() if allocation is not None else None,
            slope_migration="slope_migration.json" if slope_migration else None,
            pit=command.pit.snapshot() if args.mixed_pit else None,
            reward_weights={name: raw.reward_manager.get_term_cfg(name).weight
                            for name in ("lin_vel_z_l2", "base_height_l2")},
        )
        (args.output / "run.json").write_text(json.dumps(run_receipt, indent=2))
        print("RUN_RECEIPT", json.dumps(run_receipt), flush=True)
        del saved
        original_log = runner.logger.log

        def log(**kwargs):
            original_log(**kwargs)
            if allocation is not None:
                decision = allocation.check(kwargs["it"] + 1)
                if decision is not None:
                    with (args.output/"allocation_decisions.jsonl").open("a") as f:
                        f.write(json.dumps(decision)+"\n")
            if not all(math.isfinite(float(x)) for x in kwargs["loss_dict"].values()):
                raise FloatingPointError("Non-finite training loss")
            height_weight = raw.reward_manager.get_term_cfg("base_height_l2").weight
            if height_penalty == "off" and height_weight != 0:
                raise RuntimeError("Disabled base height penalty was re-enabled")
            row = dict(state="training", completed_updates=kwargs["it"] + 1,
                       seconds=kwargs["collect_time"] + kwargs["learn_time"],
                       collect_seconds=kwargs["collect_time"], learn_seconds=kwargs["learn_time"],
                       fast_delay=args.fast_delay,
                       actuator_profile=actuator_profile,
                       actuator_guard=guard.snapshot() if guard is not None else None,
                       command_curriculum_start=command_start,
                       terrain_training_profile=terrain_profile,
                       stair_command_profile=stair_profile,
                       wave_reset_profile=wave_profile, wave_reset=reset_snapshot(raw),
                       stair_speed=command.snapshot() if stair_profile != "original" else None,
                       base_height_penalty_weight=height_weight,
                       mixed_profile="mixed9_v1" if args.mixed_pit else None,
                       slope_profile=slope_profile,
                       pit_training_profile=pit_profile,
                       pit_posture_profile=posture_profile, pit_posture=posture_snapshot(raw),
                       surface_posture_profile=surface_profile, surface_posture=surface_snapshot(raw),
                       stance_strength_profile=stance_strength, stance_strength=stance_snapshot(raw),
                       pit_course_profile=pit_course_profile,
                       pit=command.pit.snapshot() if args.mixed_pit else None,
                       allocation_profile=allocation_profile,
                       terrain_allocation=allocation.snapshot() if allocation is not None else None,
                       terrain_by_type=summarize_terrain(terrain.terrain_levels.cpu().tolist(),
                                                        terrain.terrain_types.cpu().tolist(), column_names, slope_profile),
                       command_ranges=command.command_ranges,
                       flat_speed=flat_speed_snapshot(command),
                       uphill_speed=uphill_speed_snapshot(command),
                       observed_command_abs_max=command.command.abs().amax(dim=0).cpu().tolist(),
                       terrain=float(raw.scene.terrain.terrain_levels[
                           ~command.pit.active].float().mean()) if args.mixed_pit else float(
                               raw.scene.terrain.terrain_levels.float().mean()),
                       **kwargs["loss_dict"])
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            (args.output / "status.json").write_text(json.dumps(row, indent=2))
            if guard is not None:
                if any(j["torque_violation_samples"] for j in row["actuator_guard"]["joints"].values()):
                    raise RuntimeError("Applied torque exceeded the guarded budget")
                guard.reset_statistics()
            if (args.output / "STOP").exists():
                runner.save(str(args.output / f"model_{kwargs['it']}.pt"), kwargs["it"], True)
                raise InterruptedError("STOP requested after saving")

        runner.logger.log = log
        # Straight traversals need real elapsed episode time for their distance budget.
        runner.learn(args.updates, init_at_random_ep_len=stair_profile == "original")
        result = dict(state="complete", completed_updates=runner.current_learning_iteration + 1)
        (args.output / "SUCCESS.json").write_text(json.dumps(result))
        (args.output / "status.json").write_text(json.dumps(result))
        env.close()
    except BaseException as exc:
        traceback.print_exc()
        (args.output / "status.json").write_text(json.dumps(dict(state="stopped" if isinstance(exc, InterruptedError) else "failed", error=str(exc))))
        # Isaac shutdown can hang on a failed environment; preserve traceback then exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
