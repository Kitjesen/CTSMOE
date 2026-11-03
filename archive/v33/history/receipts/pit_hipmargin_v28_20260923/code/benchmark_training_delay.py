"""Isolated ABBA timing of both buffers in one 4096-env training process.

This benchmark discards its learned weights. Never use its output as a training
checkpoint. The two paths share the same buffers, actuator models and reset RNG.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-tail", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.root.resolve()
    sys.path[:0] = [str(root / "upstream/source/rsl_rl"), str(root / "upstream/source/robot_lab"),
                   str(root / "thunder/source/doso_train")]
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        import torch
        import cProfile
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from robot_lab.tasks.go2.env.go2_env import Go2Env
        from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg
        from rsl_rl.runners import OnPolicyRunnerCTS
        from config import make_config
        from fast_delay import FastDelayBuffer

        torch.set_num_threads(4)
        mode = {"value": "reference"}
        compute_calls = {"reference": 0, "fast": 0}
        fast_compute = FastDelayBuffer.compute

        def measured_compute(self, data):
            compute_calls[mode["value"]] += 1
            if mode["value"] == "fast":
                return fast_compute(self, data)
            result = self._reference.compute(data)
            # The reference append has consumed any pending reset notifications.
            self._pending_all = False
            self._pending_resets.clear()
            return result

        FastDelayBuffer.compute = measured_compute
        cfg = make_config(args.num_envs, root / "cache/usd", fast_delay=True)
        cfg.seed = 42
        cfg.log_dir = str(args.output)
        raw = Go2Env(cfg=cfg)
        agent = MoECTSRunnerCfg()
        agent.seed = 42
        agent.logger = "tensorboard"
        agent.device = "cuda:0"
        agent_dict = agent.to_dict()
        agent_dict["robogauge"] = {"enabled": False}
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        actuators = raw.scene["robot"].actuators
        assert set(actuators) == {"hip", "thigh", "calf", "wheel"}
        for actuator in actuators.values():
            assert isinstance(actuator.positions_delay_buffer, FastDelayBuffer)

        class BenchmarkRunner(OnPolicyRunnerCTS):
            def save(self, *args, **kwargs):
                pass  # This isolated A/B experiment must never become a checkpoint.

        runner = BenchmarkRunner(env, agent_dict, log_dir=str(args.output), device="cuda:0")
        runner.load(str(args.resume))
        saved = torch.load(args.resume, map_location="cuda:0", weights_only=False)
        runner.current_learning_iteration = saved["completed_updates"]
        runner.alg.learning_rate = saved["learning_rate"]
        raw.common_step_counter = saved["common_step_counter"]
        terrain = raw.scene.terrain
        terrain.terrain_levels[:] = saved["terrain_levels"]
        terrain.terrain_types[:] = saved["terrain_types"]
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        raw.scene.env_origins[:] = terrain.env_origins
        for name in ("base_linear_velocity", "base_height_l2"):
            term = getattr(raw.cfg.curriculum, name)
            term.func(raw, slice(None), **term.params)
        compute = raw.curriculum_manager.compute
        raw.curriculum_manager.compute = lambda env_ids: None
        try:
            raw.reset()
        finally:
            raw.curriculum_manager.compute = compute

        order = ["reference", "fast", "fast", "reference"] * 2
        total = args.warmup + args.block_size * len(order)
        rows = []
        original_log = runner.logger.log
        original_returns = runner.alg.compute_returns
        clocks = {}

        def compute_returns(obs):
            # The upstream phase timers do not join outstanding CUDA work.
            torch.cuda.synchronize()
            clocks["collect_finished"] = time.perf_counter()
            return original_returns(obs)

        runner.alg.compute_returns = compute_returns
        torch.cuda.synchronize()
        clocks["started"] = time.perf_counter()
        profiler = cProfile.Profile() if args.profile else None
        if profiler:
            profiler.enable()

        def log(**kwargs):
            torch.cuda.synchronize()
            finished = time.perf_counter()
            original_log(**kwargs)
            assert all(math.isfinite(float(v)) for v in kwargs["loss_dict"].values())
            index = len(rows)
            row = dict(index=index, mode=mode["value"], warmup=index < args.warmup,
                       collect_seconds=kwargs["collect_time"], learn_seconds=kwargs["learn_time"],
                       seconds=kwargs["collect_time"] + kwargs["learn_time"],
                       wall_seconds=finished - clocks["started"],
                       wall_collect_seconds=clocks["collect_finished"] - clocks["started"],
                       wall_learn_seconds=finished - clocks["collect_finished"],
                       terrain=float(terrain.terrain_levels.float().mean()),
                       block=(index - args.warmup) // args.block_size)
            rows.append(row)
            if profiler:
                profiler.disable()
                profiler.dump_stats(str(args.output / f"profile_{index}_{mode['value']}.prof"))
                profiler.clear()
            with (args.output / "timings.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            next_index = index + 1 - args.warmup
            if 0 <= next_index < len(order) * args.block_size:
                mode["value"] = order[next_index // args.block_size]
            if profiler:
                profiler.enable()
            torch.cuda.synchronize()
            clocks["started"] = time.perf_counter()

        runner.logger.log = log
        runner.learn(total, init_at_random_ep_len=True)
        summary = dict(num_envs=args.num_envs, resume=str(args.resume), warmup=args.warmup,
                       block_size=args.block_size, order=order, samples_per_mode=args.block_size * 4,
                       compute_calls=compute_calls)
        for field in ("collect_seconds", "learn_seconds", "seconds", "wall_collect_seconds",
                      "wall_learn_seconds", "wall_seconds"):
            values = {m: [r[field] for r in rows if not r["warmup"] and r["mode"] == m]
                      for m in ("reference", "fast")}
            medians = {m: statistics.median(v) for m, v in values.items()}
            summary[field] = dict(medians=medians,
                                  reduction=1 - medians["fast"] / medians["reference"],
                                  speedup=medians["reference"] / medians["fast"])
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("BENCHMARK_OK", json.dumps(summary), flush=True)
        if args.profile_tail:
            # Keep cProfile overhead outside the timed A/B sample set.
            profiler = cProfile.Profile()
            for path in ("reference", "fast"):
                mode["value"] = path
                torch.cuda.synchronize()
                clocks["started"] = time.perf_counter()
                profiler.enable()
                runner.learn(2, init_at_random_ep_len=False)
                profiler.disable()
        env.close()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
