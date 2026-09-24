"""Model-only migration and recoverable CTS training for the single-ledge pit task."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--warm-start", type=Path)
    source.add_argument("--resume", type=Path)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--max-level", type=int, default=4)
    p.add_argument("--probe-steps", type=int, default=0)
    p.add_argument("--support-check", action="store_true")
    p.add_argument("--save-interval", type=int, default=250)
    args = p.parse_args()
    if args.output.exists() or not 0 <= args.max_level <= 21 or args.num_envs % 4:
        raise ValueError("New output, valid level and environment multiple of four required")
    args.output.mkdir(parents=True)
    root = args.root.resolve()
    sys.path[:0] = [str(root/"upstream/source/rsl_rl"), str(root/"upstream/source/robot_lab"),
                   str(root/"thunder/source/doso_train")]
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        import torch
        from tensordict import TensorDict
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from robot_lab.tasks.go2.env.go2_env import Go2Env
        from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
        from rsl_rl.runners import OnPolicyRunnerCTS
        from config import make_config
        from actuator_guard import ActuatorGuard, SLEW_PROFILE
        from pit_task import PROFILE, configure_pit
        from pit_logic import PitCourse
        from policy_contract import checkpoint_profile, export_contract

        torch.set_num_threads(4)
        src = args.resume or args.warm_start
        saved = torch.load(src, map_location="cpu", weights_only=False)
        if checkpoint_profile(saved) != SLEW_PROFILE:
            raise ValueError("Pit experiment requires the existing v3 actuator contract")
        cfg = make_config(args.num_envs, root/"cache/usd", fast_delay=True, actuator_profile=SLEW_PROFILE)
        configure_pit(cfg, max_level=args.max_level)
        cfg.seed, cfg.log_dir = 73, str(args.output)
        agent = MoECTSRunnerCfg()
        agent.seed, agent.save_interval = 73, args.save_interval
        # New task: explicit fresh optimizer rate, not the old task's decayed rate.
        agent.algorithm.learning_rate = 3e-4
        agent.algorithm.student_encoder_learning_rate = 3e-4
        agent_dict = agent.to_dict()
        agent_dict["robogauge"] = {"enabled": False}
        dump_yaml(str(args.output/"env.yaml"), cfg)
        dump_yaml(str(args.output/"agent.yaml"), agent_dict)
        raw = Go2Env(cfg=cfg)
        guard = ActuatorGuard(raw, SLEW_PROFILE)
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        command = raw.command_manager.get_term("base_velocity")

        class StudentExport(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.encoder = copy.deepcopy(policy.student_moe_encoder).cpu().eval()
                self.actor = copy.deepcopy(policy.actor).cpu().eval()

            def forward(self, history, single_obs):
                latent, _ = self.encoder(history)
                return self.actor(torch.cat((latent, single_obs), -1))

        class PitRunner(OnPolicyRunnerCTS):
            def save(self, path, it, last_model, infos=None):
                path = Path(path)
                super().save(str(path)+".tmp", it, last_model, infos)
                checkpoint = torch.load(str(path)+".tmp", weights_only=False)
                checkpoint.update(
                    completed_updates=it+1, common_step_counter=raw.common_step_counter,
                    learning_rate=self.alg.learning_rate, pit_profile=PROFILE,
                    pit_course=command.course.state_dict(), pit_totals=command.totals,
                    terrain_levels=raw.scene.terrain.terrain_levels,
                    terrain_types=raw.scene.terrain.terrain_types,
                    actuator_profile=SLEW_PROFILE, actuator_contract=export_contract(SLEW_PROFILE),
                    source_checkpoint=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
                torch.save(checkpoint, str(path)+".tmp")
                os.replace(str(path)+".tmp", path)
                exporter = StudentExport(self.alg.policy)
                obs = TensorDict(raw.obs_buf, batch_size=[args.num_envs])[:8]
                history, single = obs["policy"].cpu(), obs["single_obs"].cpu()
                with torch.no_grad():
                    expected = self.alg.policy.act_inference(obs).cpu()
                    torch.testing.assert_close(exporter(history, single), expected, atol=3e-5, rtol=3e-5)
                    traced = torch.jit.trace(exporter, (history, single))
                    torch.testing.assert_close(traced(history, single), expected, atol=3e-5, rtol=3e-5)
                    traced.save(str(path.with_name(f"student_{it+1}.pt")), _extra_files={
                        "actuator_contract.json": json.dumps(export_contract(SLEW_PROFILE)),
                        "task_contract.json": json.dumps({"pit_profile": PROFILE,
                                                           "stop_protocol": "external platform stop"})})
                print("PIT_SAVE_EXPORT_OK", it+1, flush=True)

        runner = PitRunner(env, agent_dict, log_dir=str(args.output), device="cuda:0")
        command.learner[runner.alg.student_env_idxs] = 1
        if args.resume:
            if saved.get("pit_profile") != PROFILE:
                raise ValueError("Use warm-start for migration from the old task")
            if saved["terrain_levels"].shape != (args.num_envs,):
                raise ValueError("Exact task resume requires the same environment count")
            runner.load(str(src), map_location="cpu")
            runner.current_learning_iteration = saved["completed_updates"]
            runner.alg.learning_rate = saved["learning_rate"]
            raw.common_step_counter = saved["common_step_counter"]
            command.course = PitCourse.restore(saved["pit_course"])
            if command.course.max_level != args.max_level:
                raise ValueError("Resume must preserve validated curriculum cap")
            command.totals.copy_(saved["pit_totals"].to(raw.device))
            for name in ("terrain_levels", "terrain_types"):
                getattr(raw.scene.terrain, name).copy_(saved[name].to(raw.device))
            torch.testing.assert_close(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"],
                                       check_device=False, rtol=0, atol=0)
            torch.testing.assert_close(runner.alg.optimizer_stu_enc.state_dict(), saved["optimizer_stu_enc_state_dict"],
                                       check_device=False, rtol=0, atol=0)
        else:
            runner.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
            assert not runner.alg.optimizer.state_dict()["state"]
            assert not runner.alg.optimizer_stu_enc.state_dict()["state"]
        torch.testing.assert_close(runner.alg.policy.state_dict(), saved["model_state_dict"],
                                   check_device=False, rtol=0, atol=0)
        if args.probe_steps:
            # Matched levels within each learner group; 15/25/35 cm true single steps.
            levels = torch.zeros(args.num_envs, dtype=torch.long, device=raw.device)
            for indices in (runner.alg.teacher_env_idxs, runner.alg.student_env_idxs):
                levels[indices] = (torch.arange(len(indices), device=raw.device) % 3)*2
            command.fixed_levels = levels
            raw.scene.terrain.terrain_levels.copy_(levels)
        terrain = raw.scene.terrain
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        raw.scene.env_origins[:] = terrain.env_origins
        compute = raw.curriculum_manager.compute
        raw.curriculum_manager.compute = lambda env_ids: None
        try:
            raw.reset()
        finally:
            raw.curriculum_manager.compute = compute
        command.last_refresh = -1
        obs = env.get_observations()
        for key, width in (("policy", 530), ("single_obs", 53), ("critic", 291)):
            assert obs[key].shape == (args.num_envs, width)
            assert torch.isfinite(obs[key]).all(), key
        scan = raw.scene["height_scanner"].data.ray_hits_w
        if not torch.isfinite(scan).all():
            raise RuntimeError("Pit teacher scan has missing mesh hits")
        receipt = dict(
            state="probe" if args.probe_steps else "training", profile=PROFILE,
            source_checkpoint=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            source_completed_updates=saved.get("completed_updates"),
            migration="resume" if args.resume else "strict model-only; new optimizers and task curriculum",
            starting_updates=runner.current_learning_iteration, additional_updates=args.updates,
            num_envs=args.num_envs, teacher_envs=len(runner.alg.teacher_env_idxs),
            student_envs=len(runner.alg.student_env_idxs), actuator_contract=export_contract(SLEW_PROFILE),
            self_collisions=True, observations={k: list(v.shape) for k, v in obs.items()},
            scanner_finite=True, pit=command.snapshot())
        write_json(args.output/"run.json", receipt)
        print("PIT_CONTRACT_OK", json.dumps({k: v for k, v in receipt.items() if k != "pit"}), flush=True)
        del saved

        if args.support_check:
            from pit_support_check import validate_support
            result = validate_support(raw, env)
            write_json(args.output/"support_check.json", result)
            write_json(args.output/"SUCCESS.json", {"state": "support_check_passed"})
        elif args.probe_steps:
            runner.eval_mode()
            with torch.inference_mode():
                for step in range(args.probe_steps):
                    actions = runner.alg.policy.act_inference(obs)
                    ids = runner.alg.teacher_env_idxs
                    policy = runner.alg.policy
                    teacher_obs = policy.critic_obs_normalizer(policy.get_critic_obs(obs[ids]))
                    actions[ids] = policy.actor(torch.cat((policy.teacher_encoder(teacher_obs),
                                                           policy.single_obs_normalizer(obs["single_obs"][ids])), -1))
                    obs, _, _, _ = env.step(actions)
                    if step % 100 == 0:
                        print("PIT_PROBE", step, flush=True)
            write_json(args.output/"probe.json", command.snapshot())
            write_json(args.output/"SUCCESS.json", {"state": "probe_complete", "steps": args.probe_steps})
        else:
            original_log = runner.logger.log

            def log(**kwargs):
                original_log(**kwargs)
                if not all(math.isfinite(float(v)) for v in kwargs["loss_dict"].values()):
                    raise FloatingPointError("Non-finite pit training loss")
                guard_snapshot = guard.snapshot()
                if any(x["torque_violation_samples"] for x in guard_snapshot["joints"].values()):
                    raise RuntimeError("Actual applied torque exceeded guard")
                row = dict(state="training", completed_updates=kwargs["it"]+1,
                           seconds=kwargs["collect_time"]+kwargs["learn_time"],
                           collect_seconds=kwargs["collect_time"], learn_seconds=kwargs["learn_time"],
                           learning_rate=runner.alg.learning_rate, pit=command.snapshot(),
                           actuator_guard=guard_snapshot, losses=kwargs["loss_dict"])
                with (args.output/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row)+"\n")
                write_json(args.output/"status.json", row)
                guard.reset_statistics()
                if (args.output/"STOP").exists():
                    runner.save(args.output/f"model_{kwargs['it']}.pt", kwargs["it"], True)
                    raise InterruptedError("Saved on STOP")

            runner.logger.log = log
            runner.learn(args.updates, init_at_random_ep_len=False)
            write_json(args.output/"SUCCESS.json", dict(state="complete", completed_updates=runner.current_learning_iteration+1))
            write_json(args.output/"status.json", dict(state="complete", completed_updates=runner.current_learning_iteration+1))
        env.close()
    except BaseException as exc:
        traceback.print_exc()
        write_json(args.output/"status.json", dict(state="stopped" if isinstance(exc, InterruptedError) else "failed", error=str(exc)))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
