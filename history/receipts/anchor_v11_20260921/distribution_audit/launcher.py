"""Launch a zero-update audit through the exact V11 training setup."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import traceback
from typing import Any


EXPECTED_COMPLETED_UPDATES = 44_759
NUM_ENVS = 4_096
HORIZONS = 20
STEPS_PER_HORIZON = 24
SMOKE_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Three API steps; not a distribution audit")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_value(digest: Any, value: Any, torch: Any) -> None:
    """Hash nested state without depending on torch.save container metadata."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=repr):
            _hash_value(digest, key, torch)
            _hash_value(digest, value[key], torch)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _hash_value(digest, item, torch)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode() + b"\0" + repr(value).encode())
    else:
        raise TypeError(f"Unsupported state value for hashing: {type(value)!r}")


def state_sha256(state: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, state, torch)
    return digest.hexdigest()


def load_stats_class(path: Path) -> type:
    if not path.is_file():
        raise FileNotFoundError(f"Missing audit aggregation module: {path}")
    spec = importlib.util.spec_from_file_location("distribution_audit_stats", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load audit aggregation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats_class = getattr(module, "Stats", None)
    if not isinstance(stats_class, type):
        raise TypeError("audit_stats.py must define a Stats class")
    for name in ("before", "after", "finish"):
        if not callable(getattr(stats_class, name, None)):
            raise TypeError(f"Stats.{name} must be callable")
    return stats_class


class MutationGuard:
    """Replace every forbidden mutation entry point and restore it on exit."""

    def __init__(self, runner: Any):
        self.runner = runner
        self.calls: list[str] = []
        self._originals: list[tuple[Any, str, Any]] = []
        self.restored = False

    def _forbid(self, owner: Any, attribute: str, label: str) -> None:
        original = getattr(owner, attribute)

        def forbidden(*_args: Any, **_kwargs: Any) -> None:
            self.calls.append(label)
            raise RuntimeError(f"Forbidden during zero-update audit: {label}")

        self._originals.append((owner, attribute, original))
        setattr(owner, attribute, forbidden)

    def __enter__(self) -> MutationGuard:
        alg = self.runner.alg
        try:
            self._forbid(alg, "update", "alg.update")
            self._forbid(alg, "compute_returns", "alg.compute_returns")
            self._forbid(alg.optimizer, "step", "alg.optimizer.step")
            self._forbid(alg.optimizer_stu_enc, "step", "alg.optimizer_stu_enc.step")
            self._forbid(self.runner, "save", "runner.save")
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        for owner, attribute, original in reversed(self._originals):
            setattr(owner, attribute, original)
        self.restored = all(getattr(owner, attribute) is original for owner, attribute, original in self._originals)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._restore()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_audit_learn(
    *,
    torch: Any,
    stats_path: Path,
    output: Path,
    reference_path: Path,
    checkpoint_completed: int,
    smoke: bool,
):
    """Create the inherited ReceiptRunner.learn replacement."""

    def audit_learn(self: Any, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        status_path = output / "audit_status.json"
        before_hashes: dict[str, str] = {}
        guard: MutationGuard | None = None
        stats: Any | None = None
        stats_closed = False
        total_steps = SMOKE_STEPS if smoke else HORIZONS * STEPS_PER_HORIZON
        try:
            if type(self).__name__ != "ReceiptRunner" or "learn" in type(self).__dict__:
                raise TypeError("Expected ReceiptRunner to inherit the patched OnPolicyRunnerCTS.learn")
            if num_learning_iterations != 0:
                raise RuntimeError(f"Audit requires --updates 0, got {num_learning_iterations}")
            if self.env.num_envs != NUM_ENVS:
                raise RuntimeError(f"Audit requires {NUM_ENVS} environments, got {self.env.num_envs}")
            if self.cfg["num_steps_per_env"] != STEPS_PER_HORIZON:
                raise RuntimeError(
                    f"Audit requires {STEPS_PER_HORIZON} native rollout steps, got {self.cfg['num_steps_per_env']}"
                )
            if self.current_learning_iteration != checkpoint_completed:
                raise RuntimeError(
                    "Runner did not retain checkpoint completed_updates: "
                    f"{self.current_learning_iteration} != {checkpoint_completed}"
                )
            policy = self.alg.policy
            if policy.actor_obs_normalization or policy.critic_obs_normalization:
                raise RuntimeError("V11 audit expects actor and critic observation normalization to be disabled")
            if self.alg.rnd is not None:
                raise RuntimeError("Unexpected RND normalizer in the V11 training configuration")
            raw = self.env.unwrapped
            if raw is self.env or raw.num_envs != NUM_ENVS:
                raise RuntimeError("runner.env.unwrapped did not resolve to the 4096-env Go2 environment")
            if self.alg.storage.step != 0:
                raise RuntimeError(f"Rollout storage was not empty at audit start: {self.alg.storage.step}")

            reference_state = torch.load(reference_path, map_location="cpu", weights_only=False)
            reference_policy = copy.deepcopy(policy)
            reference_policy.load_state_dict(reference_state["model_state_dict"], strict=True)
            reference_policy.to(self.device).eval().requires_grad_(False)
            del reference_state
            stats_class = load_stats_class(stats_path)

            before_hashes = {
                "model": state_sha256(policy.state_dict(), torch),
                "ppo_optimizer": state_sha256(self.alg.optimizer.state_dict(), torch),
                "student_optimizer": state_sha256(self.alg.optimizer_stu_enc.state_dict(), torch),
            }
            write_json(
                status_path,
                {
                    "state": "audit_collecting",
                    "smoke_only": smoke,
                    "completed_updates": checkpoint_completed,
                    "optimizer_updates": 0,
                    "planned_control_steps": total_steps,
                },
            )
            stats = stats_class(raw, self.alg, reference_policy, output)
            if init_at_random_ep_len:
                self.env.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )
            obs = self.env.get_observations().to(self.device)
            self.train_mode()
            if self.is_distributed:
                raise RuntimeError("Distribution audit only supports the single-process V11 run")

            guard = MutationGuard(self)
            with guard, torch.inference_mode():
                horizon_lengths = [SMOKE_STEPS] if smoke else [STEPS_PER_HORIZON] * HORIZONS
                collected = 0
                for horizon, horizon_steps in enumerate(horizon_lengths):
                    if self.alg.storage.step != 0:
                        raise RuntimeError(f"Storage nonempty before horizon {horizon}: {self.alg.storage.step}")
                    for _ in range(horizon_steps):
                        actions = self.alg.act(obs)
                        stats.before(obs, actions)
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        obs, rewards, dones = (
                            obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                        )
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        stats.after(obs, rewards, dones, extras)
                        collected += 1
                    if self.alg.storage.step != horizon_steps:
                        raise RuntimeError(
                            f"Unexpected storage step after horizon {horizon}: "
                            f"{self.alg.storage.step} != {horizon_steps}"
                        )
                    self.alg.storage.clear()
                if collected != total_steps:
                    raise RuntimeError(f"Collected {collected} control steps, expected {total_steps}")
                stats.finish()
                stats.close()
                stats_closed = True

            if not guard.restored:
                raise RuntimeError("Forbidden mutation patches were not restored")
            after_hashes = {
                "model": state_sha256(policy.state_dict(), torch),
                "ppo_optimizer": state_sha256(self.alg.optimizer.state_dict(), torch),
                "student_optimizer": state_sha256(self.alg.optimizer_stu_enc.state_dict(), torch),
            }
            if before_hashes != after_hashes:
                raise RuntimeError(f"Model or optimizer state changed: {before_hashes} != {after_hashes}")
            if guard.calls:
                raise RuntimeError(f"Forbidden mutation entry points were called: {guard.calls}")
            if self.alg.storage.step != 0:
                raise RuntimeError(f"Storage was not cleared after audit: {self.alg.storage.step}")
            write_json(
                status_path,
                {
                    "state": "AUDIT_COMPLETE" if not smoke else "AUDIT_SMOKE_COMPLETE",
                    "authoritative": True,
                    "smoke_only": smoke,
                    "completed_updates": checkpoint_completed,
                    "optimizer_updates": 0,
                    "horizons": 1 if smoke else HORIZONS,
                    "control_steps": total_steps,
                    "num_envs": NUM_ENVS,
                    "normalization_disabled": True,
                    "storage_cleared_each_horizon": True,
                    "mutation_patches_restored": True,
                    "stats_environment_patch_restored": stats_closed,
                    "forbidden_calls": [],
                    "state_hashes_before": before_hashes,
                    "state_hashes_after": after_hashes,
                },
            )
        except BaseException as exc:
            if stats is not None and callable(getattr(stats, "close", None)):
                stats.close()
                stats_closed = True
            write_json(
                status_path,
                {
                    "state": "AUDIT_FAILED",
                    "authoritative": True,
                    "smoke_only": smoke,
                    "completed_updates": checkpoint_completed,
                    "optimizer_updates": 0,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "mutation_patches_restored": bool(guard and guard.restored),
                    "stats_environment_patch_restored": stats_closed,
                    "forbidden_calls": [] if guard is None else guard.calls,
                    "state_hashes_before": before_hashes,
                },
            )
            raise

    return audit_learn


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    code = args.code.resolve()
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    reference = args.reference.resolve()
    train_path = code / "train.py"
    stats_path = Path(__file__).resolve().parent / "audit_stats.py"
    if output.exists():
        raise FileExistsError(f"Audit output must not exist: {output}")
    for label, path in (("train entry", train_path), ("checkpoint", checkpoint), ("reference", reference)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if checkpoint == reference:
        raise ValueError("V11 checkpoint and V6 reference must be distinct files")

    sys.path[:0] = [str(code), str(root / "upstream/source/rsl_rl")]
    import torch
    from rsl_rl.runners import OnPolicyRunnerCTS

    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_completed = int(checkpoint_state["completed_updates"])
    del checkpoint_state
    if checkpoint_completed != EXPECTED_COMPLETED_UPDATES:
        raise ValueError(
            f"Stage 0 is fixed to V11 {EXPECTED_COMPLETED_UPDATES}, got {checkpoint_completed}"
        )
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing audit aggregation module: {stats_path}")
    checkpoint_sha_before = file_sha256(checkpoint)
    reference_sha_before = file_sha256(reference)
    audit_learn = make_audit_learn(
        torch=torch,
        stats_path=stats_path,
        output=output,
        reference_path=reference,
        checkpoint_completed=checkpoint_completed,
        smoke=args.smoke,
    )
    original_learn = OnPolicyRunnerCTS.learn
    original_argv = sys.argv[:]
    OnPolicyRunnerCTS.learn = audit_learn
    sys.argv = [
        str(train_path),
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
        "--startup-anchor-reference",
        str(reference),
    ]
    try:
        runpy.run_path(str(train_path), run_name="__main__")
    finally:
        sys.argv = original_argv
        OnPolicyRunnerCTS.learn = original_learn

    audit_status_path = output / "audit_status.json"
    audit_status = json.loads(audit_status_path.read_text(encoding="utf-8"))
    expected_state = "AUDIT_SMOKE_COMPLETE" if args.smoke else "AUDIT_COMPLETE"
    if audit_status.get("state") != expected_state or audit_status.get("optimizer_updates") != 0:
        raise RuntimeError(f"Audit did not complete cleanly: {audit_status}")
    checkpoint_sha_after = file_sha256(checkpoint)
    reference_sha_after = file_sha256(reference)
    if checkpoint_sha_before != checkpoint_sha_after or reference_sha_before != reference_sha_after:
        raise RuntimeError("Checkpoint or reference file changed during the read-only audit")
    checkpoint_artifacts = sorted(output.glob("model_*.pt")) + sorted(output.glob("student_*.pt"))
    checkpoint_artifacts += sorted(output.glob("*.tmp"))
    if checkpoint_artifacts:
        raise RuntimeError(f"Audit unexpectedly wrote model artifacts: {checkpoint_artifacts}")
    audit_status.update(
        launcher_patch_restored=OnPolicyRunnerCTS.learn is original_learn,
        checkpoint_sha256=checkpoint_sha_after,
        reference_sha256=reference_sha_after,
        input_files_unchanged=True,
        checkpoint_artifacts_written=[],
        training_success_receipt_superseded=True,
    )
    write_json(audit_status_path, audit_status)
    print(
        json.dumps(
            {
                "state": "audit_requires_external_finalization",
                "completed_updates": checkpoint_completed,
                "optimizer_updates": 0,
                "authoritative_receipt": "audit_status.json",
                "finalizer": "finalize_audit.py",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
