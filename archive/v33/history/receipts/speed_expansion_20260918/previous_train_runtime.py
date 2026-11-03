"""Official Go2 CTS learning loop with isolated Thunder configuration and receipts."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--resume", type=Path)
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
        from rsl_rl.runners import OnPolicyRunnerCTS
        from config import make_config

        torch.set_num_threads(4)
        cfg = make_config(args.num_envs, root / "cache/usd")
        cfg.seed = 42
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
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        obs = env.get_observations()
        dimensions = {key: list(value.shape) for key, value in obs.items()}
        assert obs["policy"].shape == (args.num_envs, 530), dimensions
        assert obs["single_obs"].shape == (args.num_envs, 53), dimensions
        assert obs["critic"].shape == (args.num_envs, 291), dimensions
        assert env.num_actions == 16
        assert raw.scene.terrain.terrain_origins.shape == (10, 20, 3)
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
                             common_step_counter=raw.common_step_counter,
                             terrain_levels=raw.scene.terrain.terrain_levels,
                             terrain_types=raw.scene.terrain.terrain_types,
                             learning_rate=self.alg.learning_rate)
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
                    traced.save(str(p.with_name(f"student_{it + 1}.pt")))
                print("SAVE_EXPORT_OK", it + 1, flush=True)

        runner = ReceiptRunner(env, agent_dict, log_dir=str(args.output), device="cuda:0")
        if args.resume:
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
            compute = raw.curriculum_manager.compute
            raw.curriculum_manager.compute = lambda env_ids: None
            raw.reset()
            raw.curriculum_manager.compute = compute
            print("RESUME_OK", runner.current_learning_iteration, flush=True)
        original_log = runner.logger.log

        def log(**kwargs):
            original_log(**kwargs)
            if not all(math.isfinite(float(x)) for x in kwargs["loss_dict"].values()):
                raise FloatingPointError("Non-finite training loss")
            row = dict(state="training", completed_updates=kwargs["it"] + 1,
                       seconds=kwargs["collect_time"] + kwargs["learn_time"],
                       terrain=float(raw.scene.terrain.terrain_levels.float().mean()),
                       **kwargs["loss_dict"])
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            (args.output / "status.json").write_text(json.dumps(row, indent=2))
            if (args.output / "STOP").exists():
                runner.save(str(args.output / f"model_{kwargs['it']}.pt"), kwargs["it"], True)
                raise InterruptedError("STOP requested after saving")

        runner.logger.log = log
        runner.learn(args.updates, init_at_random_ep_len=True)
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
