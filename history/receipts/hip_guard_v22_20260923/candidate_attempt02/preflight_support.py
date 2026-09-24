"""Run a 3000-step V14 preflight through train.py without optimizer updates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import traceback
from typing import Any

NUM_ENVS = 4096
CONTROL_STEPS = 3000
STEPS_PER_HORIZON = 24
PROFILE = "early_phase_v14"
ALLOWED_COMPLETED_UPDATES = {30000, 31831}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_value(digest: Any, value: Any, torch: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("Non-finite tensor in model/optimizer state")
        digest.update(str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            _hash_value(digest, key, torch)
            _hash_value(digest, value[key], torch)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _hash_value(digest, item, torch)
    elif value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not __import__("math").isfinite(value):
            raise RuntimeError("Non-finite scalar in model/optimizer state")
        digest.update(type(value).__name__.encode() + repr(value).encode())
    else:
        raise TypeError(f"Unsupported hash value: {type(value)!r}")


def state_sha256(state: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, state, torch)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class MutationGuard:
    def __init__(self, runner: Any):
        self.runner, self.calls, self.originals = runner, [], []
        self.restored = False

    def _forbid(self, owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)

        def forbidden(*_args, **_kwargs):
            self.calls.append(label)
            raise RuntimeError(f"Forbidden call: {label}")

        self.originals.append((owner, name, original))
        setattr(owner, name, forbidden)

    def __enter__(self):
        for owner, name, label in (
            (self.runner.alg, "update", "alg.update"),
            (self.runner.alg, "compute_returns", "alg.compute_returns"),
            (self.runner.alg.optimizer, "step", "optimizer.step"),
            (self.runner.alg.optimizer_stu_enc, "step", "student_optimizer.step"),
            (self.runner, "save", "runner.save"),
        ):
            self._forbid(owner, name, label)
        return self

    def __exit__(self, *_args):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.restored = all(
            getattr(owner, name) is original for owner, name, original in self.originals
        )


def load_stats(path: Path):
    spec = importlib.util.spec_from_file_location("early_v14_audit_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Stats


def make_audit_learn(
    torch,
    stats_path: Path,
    output: Path,
    source_completed: int,
    checkpoint: Path,
    checkpoint_hash: str,
):
    def audit_learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ):
        status_path, guard, stats = output / "audit_status.json", None, None
        before = {}
        try:
            if (
                num_learning_iterations != 0
                or self.current_learning_iteration != source_completed
            ):
                raise RuntimeError(
                    "Runner iteration does not match explicit --source-completed"
                )
            if (
                self.env.num_envs != NUM_ENVS
                or self.cfg["num_steps_per_env"] != STEPS_PER_HORIZON
            ):
                raise RuntimeError("Preflight requires 4096 envs and 24-step storage")
            if init_at_random_ep_len or self.alg.storage.step != 0:
                raise RuntimeError(
                    "Preflight requires episode age zero and empty storage"
                )
            before = {
                "model": state_sha256(self.alg.policy.state_dict(), torch),
                "ppo_optimizer": state_sha256(self.alg.optimizer.state_dict(), torch),
                "student_optimizer": state_sha256(
                    self.alg.optimizer_stu_enc.state_dict(), torch
                ),
            }
            stats = load_stats(stats_path)(self.env.unwrapped, self.alg, output)
            obs = self.env.get_observations().to(self.device)
            self.train_mode()
            guard = MutationGuard(self)
            collected = 0
            with guard, torch.inference_mode():
                while collected < CONTROL_STEPS:
                    horizon = min(STEPS_PER_HORIZON, CONTROL_STEPS - collected)
                    for _ in range(horizon):
                        actions = self.alg.act(obs)
                        stats.before(obs, actions)
                        obs, rewards, dones, extras = self.env.step(
                            actions.to(self.env.device)
                        )
                        obs, rewards, dones = (
                            obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                        )
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        stats.after(obs, rewards, dones, extras)
                        collected += 1
                    if self.alg.storage.step != horizon:
                        raise RuntimeError("Unexpected rollout storage step")
                    self.alg.storage.clear()
                summary = stats.finish()
                stats = None
            after = {
                "model": state_sha256(self.alg.policy.state_dict(), torch),
                "ppo_optimizer": state_sha256(self.alg.optimizer.state_dict(), torch),
                "student_optimizer": state_sha256(
                    self.alg.optimizer_stu_enc.state_dict(), torch
                ),
            }
            if before != after or guard.calls or not guard.restored:
                raise RuntimeError("State mutation invariant failed")
            write_json(
                status_path,
                {
                    "state": "AUDIT_COMPLETE",
                    "written_before_learn_return": True,
                    "completed_updates": source_completed,
                    "optimizer_updates": 0,
                    "control_steps": CONTROL_STEPS,
                    "simulated_seconds": summary["simulated_seconds"],
                    "num_envs": NUM_ENVS,
                    "profile": PROFILE,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "state_hashes_before": before,
                    "state_hashes_after": after,
                    "forbidden_calls": [],
                    "checkpoint_artifacts_written": [],
                },
            )
        except BaseException as exc:
            if stats is not None:
                stats.close()
            write_json(
                status_path,
                {
                    "state": "AUDIT_FAILED",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "state_hashes_before": before,
                    "forbidden_calls": [] if guard is None else guard.calls,
                },
            )
            raise

    return audit_learn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "code", "output", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--source-completed",
        type=int,
        choices=sorted(ALLOWED_COMPLETED_UPDATES),
        required=True,
    )
    args = parser.parse_args()
    root, code, output = args.root.resolve(), args.code.resolve(), args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    if output.exists():
        raise FileExistsError(f"Output must be new: {output}")
    if not (code / "train.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("train.py/checkpoint")
    sys.path[:0] = [str(code), str(root / "upstream/source/rsl_rl")]
    import torch
    from rsl_rl.runners import OnPolicyRunnerCTS

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        "completed_updates" in state
        and int(state["completed_updates"]) != args.source_completed
    ):
        raise ValueError(
            "Checkpoint metadata disagrees with explicit --source-completed"
        )
    del state
    checkpoint_hash = file_sha256(checkpoint)
    original_learn, original_argv = OnPolicyRunnerCTS.learn, sys.argv[:]
    OnPolicyRunnerCTS.learn = make_audit_learn(
        torch,
        Path(__file__).with_name("audit_stats.py"),
        output,
        args.source_completed,
        checkpoint,
        checkpoint_hash,
    )
    sys.argv = [
        str(code / "train.py"),
        "--root",
        str(root),
        "--output",
        str(output),
        "--num-envs",
        str(NUM_ENVS),
        "--updates",
        "0",
        "--mixed-pit",
        "--resume",
        str(checkpoint),
        "--early-restart-profile",
        PROFILE,
        "--startup-repair-profile",
        "off",
        "--pit-posture-profile",
        "off",
        "--surface-posture-profile",
        "off",
        "--stance-strength-profile",
        "off",
        "--allocation-profile",
        "balanced_adaptive_v1",
        "--pit-training-profile",
        "strict_v1",
        "--pit-course-profile",
        "global_v1",
        "--actuator-profile",
        "rs0402_transient_v4",
        "--slope-profile",
        "slope45_v1",
        "--command-curriculum-start",
        "intermediate",
        "--base-height-penalty",
        "off",
        "--terrain-training-profile",
        "stairs80_v1",
        "--stair-command-profile",
        "straight_speed_v1",
        "--wave-reset-profile",
        "wave_surface_v1",
    ]
    try:
        runpy.run_path(str(code / "train.py"), run_name="__main__")
    finally:
        sys.argv = original_argv
        OnPolicyRunnerCTS.learn = original_learn
    status = json.loads((output / "audit_status.json").read_text(encoding="utf-8"))
    if status.get("state") != "AUDIT_COMPLETE":
        raise RuntimeError(status)
    status.update(
        input_file_unchanged=file_sha256(checkpoint) == checkpoint_hash,
        checkpoint_artifacts_written=[p.name for p in output.glob("model_*.pt")],
        launcher_patch_restored=OnPolicyRunnerCTS.learn is original_learn,
    )
    write_json(output / "audit_status.json", status)


if __name__ == "__main__":
    main()
