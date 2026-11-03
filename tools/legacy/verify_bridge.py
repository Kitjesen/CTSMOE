"""CPU-only integration check against real old weights, upstream CTS and JIT.

No physics environment and no formal training. One synthetic optimizer update
is performed only with --exercise-update; its checkpoint lives in a temporary
directory and is deleted, never counted toward the formal 20000 goal.
"""

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from thunder_lab3.training import HISTORY_LAYOUT
from thunder_lab3.training.checkpoint import SOURCE_SHA256, load_checkpoint, normalize_checkpoint, validate_for_algorithm
from thunder_lab3.training.exporter import ThunderHistoryPolicy, export_jit, export_onnx
from thunder_lab3.training.history import FrameHistory, validate_observations
from thunder_lab3.training.runner import ThunderMoeCtsRunner
from thunder_lab3.training.timeout import terminal_timeout_reward


class SyntheticRunner(ThunderMoeCtsRunner):
    """Test-only bypass of the native course gate; never use this with a simulator."""
    def _validate_formal_continuation(self):
        if not isinstance(self.env, SyntheticEnv):
            raise TypeError("Synthetic regression accepts only its fake environment")


def runner_config():
    # Pure dict avoids bringing environment configuration into this CPU check.
    return {
        "num_steps_per_env": 4, "save_interval": 100, "log_interval": 1,
        "experiment_name": "CPU_SYNTHETIC_ONLY", "logger": "tensorboard",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "newpose_compat": {"cumulative_target": 20000, "critic_dim": 309,
                           "history_layout": HISTORY_LAYOUT},
        "robogauge": {"enabled": False, "task_name": "unused", "submit_interval": 100,
                      "poll_interval": 60, "port": 9973},
        "policy": {"class_name": "rsl_rl.moe_cts:ActorCriticMoECTS", "init_noise_std": 1.,
                   "noise_std_type": "log", "actor_obs_normalization": False,
                   "critic_obs_normalization": False, "actor_hidden_dims": [512, 256, 128],
                   "critic_hidden_dims": [512, 256, 128], "teacher_encoder_hidden_dims": [512, 256],
                   "student_encoder_hidden_dims": [512, 256, 256], "expert_num": 8,
                   "latent_dim": 32, "norm_type": "l2norm", "activation": "elu"},
        "algorithm": {"class_name": "rsl_rl.moe_cts:MoECTS", "teacher_env_ratio": .75,
                      "num_learning_epochs": 2, "num_mini_batches": 2, "learning_rate": .001,
                      "student_encoder_learning_rate": .001, "symmetry_cfg": None},
    }


class SyntheticEnv:
    def __init__(self):
        self.cfg = SimpleNamespace(newpose_history_layout=HISTORY_LAYOUT,
                                   newpose_timeout_bootstrap="terminal_observation", compute_final_obs=True)
        self.device = "cpu"
        self.num_envs, self.num_actions, self.max_episode_length = 8, 16, 20
        self.episode_length_buf = torch.zeros(8, dtype=torch.long)
        self.history = FrameHistory(torch.randn(8, 53) * .1)
        self.critic = torch.randn(8, 309) * .1

    def get_observations(self):
        return self.history.pack(self.critic)

    def step(self, actions):
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[done] = 0
        frame = torch.randn(8, 53) * .1
        frame[:, -16:] = actions
        self.history.advance(frame, done)
        self.critic = torch.randn(8, 309) * .1
        return self.get_observations(), -.01 * actions.square().mean(-1), done, {}


def verify_timeout_boundary():
    from tensordict import TensorDict
    original = torch.arange(8 * 530, dtype=torch.float32).reshape(8, 530)
    teacher = torch.tensor([1, 2, 3, 5, 6, 7])
    student = torch.tensor([0, 4])
    final = {"single_obs": torch.arange(8)[:, None].repeat(1, 53).float() + 1000,
             "critic": torch.arange(8)[:, None].repeat(1, 309).float() + 2000,
             "policy": torch.full((8, 530), -999.)}

    class ValueProbe:
        def evaluate(self, obs, is_teacher):
            chosen = (obs["single_obs"][:, 0] - 1000).long()
            expected = torch.cat((original[chosen, 53:], final["single_obs"][chosen]), dim=-1)
            assert torch.equal(obs["policy"], expected)
            return (obs["single_obs"][:, 0] + obs["critic"][:, 0] + 100 * is_teacher)[:, None]

    algorithm = SimpleNamespace(policy=ValueProbe(), gamma=.99, teacher_num_envs=6,
                               teacher_env_idxs=teacher, student_env_idxs=student,
                               transition=SimpleNamespace(observations=TensorDict(
                                   {"policy": original[torch.cat((teacher, student))]}, batch_size=[8])))
    timeouts = torch.tensor([True, True, True, False, False, False, False, False])
    terminated = torch.tensor([False, False, True, False, False, False, False, False])
    extras = {"time_outs": timeouts, "final_obs": final}
    reward, clean, count = terminal_timeout_reward(algorithm, torch.zeros(8), timeouts, extras, terminated)
    expected = torch.zeros(8)
    expected[0], expected[1] = .99 * 3000, .99 * 3102
    torch.testing.assert_close(reward, expected)
    assert count == 2 and "time_outs" not in clean and "time_outs" in extras
    try:
        terminal_timeout_reward(algorithm, torch.zeros(8), timeouts, {"time_outs": timeouts}, terminated)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing final observations silently fell back to s_t bootstrap")


def equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_nested(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_nested(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--old-policy", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exercise-update", action="store_true")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    verify_timeout_boundary()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Wrong formal source checkpoint")
    old = torch.load(args.source, map_location="cpu", weights_only=False)
    saved = load_checkpoint(args.source, 309)
    env = SyntheticEnv()
    runner = SyntheticRunner(env, runner_config(), device="cpu")
    runner.load(args.source)
    validate_for_algorithm(saved, runner.alg)
    assert equal_nested(runner.alg.policy.state_dict(), saved["model_state_dict"])
    assert equal_nested(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"])
    assert equal_nested(runner.alg.optimizer_stu_enc.state_dict(), old["student_optimizer"])
    assert runner.alg.learning_rate == old["learning_rate"]
    for key in ("terrain_levels", "terrain_types", "common_step_counter", "python_rng",
                "numpy_rng", "torch_rng", "cuda_rng"):
        left, right = old[key], saved["environment_state"][key]
        if key == "numpy_rng":
            import numpy as np
            assert left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]
        else:
            assert equal_nested(left, right), key
    try:
        ThunderMoeCtsRunner._validate_formal_continuation(runner)
    except ValueError:
        pass
    else:
        raise AssertionError("Production gate accepted fresh 8-env course as formal 4096-env continuation")
    unchanged = ("student_moe_encoder.", "actor.", "log_std")
    for key, tensor in old["model"].items():
        if key.startswith(unchanged):
            assert torch.equal(tensor, runner.alg.policy.state_dict()[key]), key

    # Distinct values in every frame/feature expose term-major vs frame-major errors.
    frame = torch.arange(106, dtype=torch.float32).reshape(2, 53)
    history = FrameHistory(frame)
    for step in range(1, 12):
        next_frame = frame + 1000 * step
        history.advance(next_frame, torch.tensor([False, step == 11]))
    assert torch.equal(history.data[0, :, 0], torch.arange(2, 12) * 1000.)
    assert torch.equal(history.data[1], (frame[1] + 11000).repeat(10, 1))
    validate_observations(history.pack(torch.zeros(2, 309)))

    legacy_policy = torch.jit.load(str(args.old_policy), map_location="cpu").eval()
    deployed = ThunderHistoryPolicy(runner.alg.policy).eval()
    samples = torch.randn(32, 530) * .5
    with torch.no_grad():
        expected = legacy_policy(samples)
        actual = deployed(samples)
        torch.testing.assert_close(actual, expected, atol=1.e-6, rtol=1.e-6)
    result = {"scope": "CPU/synthetic only; no native physics or formal training",
              "source_sha256": SOURCE_SHA256, "completed_formal_updates": 19538,
              "student_actor_parameters_exact": True, "both_adam_states_exact": True,
              "history_reset_and_order_passed": True, "critic_dim": 309,
              "old_course_rng_payload_exact": True, "production_fresh_course_learn_rejected": True,
              "terminal_timeout_history_and_termination_mask_passed": True,
              "student_jit_max_error": float((actual - expected).abs().max()),
              "teacher_critic_forward_preserved": False, "algorithm": type(runner.alg).__module__}
    with tempfile.TemporaryDirectory(prefix="thunder_synthetic_") as directory:
        diagnostic = ThunderMoeCtsRunner(SyntheticEnv(), runner_config(), device="cpu")
        diagnostic.load(args.source, load_optimizer=False)
        assert diagnostic.environment_resume_status["mode"] == "diagnostic_fresh_course"
        diagnostic_file = Path(directory) / "diagnostic.pt"
        try:
            diagnostic.save(diagnostic_file)
        except ValueError:
            pass
        else:
            raise AssertionError("Diagnostic model-only load wrote a continuation checkpoint")
        assert not diagnostic_file.exists() and not Path(str(diagnostic_file) + ".tmp").exists()
        result["diagnostic_only_checkpoint_write_rejected_without_output"] = True
        del diagnostic
        export = export_jit(runner.alg.policy, directory)
        with torch.no_grad():
            serialized = torch.jit.load(str(export))(samples)
        torch.testing.assert_close(serialized, expected, atol=1.e-6, rtol=1.e-6)
        result["exported_jit_max_error"] = float((serialized - expected).abs().max())
        if args.onnx:
            output = export_onnx(runner.alg.policy, directory)
            import onnxruntime as ort
            session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            onnx_result = torch.from_numpy(session.run(None, {"obs": samples[:1].numpy()})[0])
            torch.testing.assert_close(onnx_result, expected[:1], atol=2.e-5, rtol=2.e-5)
            result["onnx_max_error"] = float((onnx_result - expected[:1]).abs().max())
        try:
            runner.learn(463)
        except ValueError:
            result["over_budget_rejected"] = True
        else:
            raise AssertionError("Cumulative target was not enforced")
        if args.exercise_update:
            before = [group["params"][0] for group in runner.alg.optimizer.param_groups]
            ppo_steps = [int(runner.alg.optimizer.state[p]["step"]) for p in before]
            student = runner.alg.optimizer_stu_enc.param_groups[0]["params"][0]
            student_step = int(runner.alg.optimizer_stu_enc.state[student]["step"])
            losses = []
            original_log = runner.logger.log

            def record_loss(*values, **named):
                losses.append(dict(named["loss_dict"]))
                return original_log(*values, **named)

            runner.logger.log = record_loss
            runner.learn(1)
            assert len(losses) == 1 and all(math.isfinite(v) for v in losses[0].values())
            assert runner.completed_updates == runner.current_learning_iteration == 19539
            assert all(int(runner.alg.optimizer.state[p]["step"]) == previous + 4
                       for p, previous in zip(before, ppo_steps))
            assert int(runner.alg.optimizer_stu_enc.state[student]["step"]) == student_step + 4
            assert all(torch.isfinite(tensor).all() for tensor in runner.alg.policy.state_dict().values())
            runner.save(Path(directory) / "model_19538.pt")
            snapshot = Path(directory) / "model_19539.pt"
            assert snapshot.exists() and not (Path(directory) / "model_19538.pt").exists()
            restored = SyntheticRunner(SyntheticEnv(), runner_config(), device="cpu")
            restored.load(snapshot)
            assert restored.completed_updates == restored.current_learning_iteration == 19539
            assert equal_nested(restored.alg.optimizer.state_dict(), runner.alg.optimizer.state_dict())
            assert equal_nested(restored.alg.optimizer_stu_enc.state_dict(), runner.alg.optimizer_stu_enc.state_dict())
            result["synthetic_update"] = {"updates": 1, "each_adam_step_delta": 4,
                                          "count_and_checkpoint_roundtrip_passed": True,
                                          "artifact_deleted": True, "formal_progress_changed": False,
                                          "losses": losses[0]}
    # Negative checks must reject schema/count guesses before mutating a runner.
    for invalid in ({"model_state_dict": {}}, {**saved, "iter": 19537}):
        try:
            normalize_checkpoint(invalid, "unknown", 309)
        except (ValueError, KeyError):
            continue
        raise AssertionError("Incompatible checkpoint accepted")
    result["incompatible_checkpoint_rejected"] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
