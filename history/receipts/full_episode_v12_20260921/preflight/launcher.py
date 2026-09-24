"""Run the V12 60-second real-course preflight with zero optimizer updates."""

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


EXPECTED_COMPLETED_UPDATES = 44_711
EXPECTED_REFERENCE_COMPLETED_UPDATES = 43_202
EXPECTED_DONOR_COMPLETED_UPDATES = 44_759
NUM_ENVS = 4_096
CONTROL_STEPS = 3_000
STEPS_PER_HORIZON = 24
SMOKE_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--coefficient-donor", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_value(digest: Any, value: Any, torch: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0" + str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            _hash_value(digest, key, torch)
            _hash_value(digest, value[key], torch)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _hash_value(digest, item, torch)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode() + b"\0" + repr(value).encode())
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
        self.runner = runner
        self.calls: list[str] = []
        self.originals: list[tuple[Any, str, Any]] = []
        self.restored = False

    def _forbid(self, owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)

        def forbidden(*_args: Any, **_kwargs: Any) -> None:
            self.calls.append(label)
            raise RuntimeError(f"Forbidden during zero-update preflight: {label}")

        self.originals.append((owner, name, original))
        setattr(owner, name, forbidden)

    def __enter__(self) -> MutationGuard:
        alg = self.runner.alg
        try:
            self._forbid(alg, "update", "alg.update")
            self._forbid(alg, "compute_returns", "alg.compute_returns")
            self._forbid(alg.optimizer, "step", "alg.optimizer.step")
            self._forbid(alg.optimizer_stu_enc, "step", "alg.optimizer_stu_enc.step")
            self._forbid(self.runner, "save", "runner.save")
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self) -> None:
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.restored = all(getattr(owner, name) is original for owner, name, original in self.originals)

    def __exit__(self, *_args: Any) -> None:
        self.restore()


def load_stats(path: Path) -> type:
    spec = importlib.util.spec_from_file_location("full_episode_audit_stats", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Stats


def make_audit_learn(
    torch: Any,
    stats_path: Path,
    output: Path,
    smoke: bool,
    checkpoint: Path,
    reference: Path,
    coefficient_donor: Path,
    checkpoint_hash: str,
    reference_hash: str,
    coefficient_donor_hash: str,
):
    total_steps = SMOKE_STEPS if smoke else CONTROL_STEPS

    def audit_learn(self: Any, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        status_path = output / "audit_status.json"
        guard: MutationGuard | None = None
        stats: Any | None = None
        before: dict[str, str] = {}
        try:
            if num_learning_iterations != 0 or self.current_learning_iteration != EXPECTED_COMPLETED_UPDATES:
                raise RuntimeError("Preflight must resume completed_updates=44711 with --updates 0")
            if self.env.num_envs != NUM_ENVS or self.cfg["num_steps_per_env"] != STEPS_PER_HORIZON:
                raise RuntimeError("Preflight requires 4096 envs and native 24-step rollout storage")
            if init_at_random_ep_len:
                raise RuntimeError("Full-episode preflight requires real episode age from zero")
            if self.alg.storage.step != 0:
                raise RuntimeError("Rollout storage was not empty at preflight start")
            raw = self.env.unwrapped
            policy = self.alg.policy
            before = {
                "model": state_sha256(policy.state_dict(), torch),
                "ppo_optimizer": state_sha256(self.alg.optimizer.state_dict(), torch),
                "student_optimizer": state_sha256(self.alg.optimizer_stu_enc.state_dict(), torch),
            }
            stats = load_stats(stats_path)(raw, self.alg, output)
            obs = self.env.get_observations().to(self.device)
            self.train_mode()  # Preserve the real training action/noise path.
            guard = MutationGuard(self)
            collected = 0
            with guard, torch.inference_mode():
                while collected < total_steps:
                    horizon_steps = min(STEPS_PER_HORIZON, total_steps - collected)
                    for _ in range(horizon_steps):
                        actions = self.alg.act(obs)
                        stats.before(obs, actions)
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        stats.after(obs, rewards, dones, extras)
                        collected += 1
                    if self.alg.storage.step != horizon_steps:
                        raise RuntimeError(f"Unexpected rollout storage step: {self.alg.storage.step}")
                    self.alg.storage.clear()
                summary = stats.finish()
                stats = None
            after = {
                "model": state_sha256(policy.state_dict(), torch),
                "ppo_optimizer": state_sha256(self.alg.optimizer.state_dict(), torch),
                "student_optimizer": state_sha256(self.alg.optimizer_stu_enc.state_dict(), torch),
            }
            if before != after or guard.calls or not guard.restored:
                raise RuntimeError("Mutation guard or state-hash invariant failed")
            state = "AUDIT_SMOKE_COMPLETE" if smoke else "AUDIT_COMPLETE"
            write_json(status_path, {
                "state": state,
                "authoritative": True,
                "written_before_learn_return": True,
                "smoke_only": smoke,
                "completed_updates": EXPECTED_COMPLETED_UPDATES,
                "optimizer_updates": 0,
                "control_steps": total_steps,
                "simulated_seconds": summary["simulated_seconds"],
                "num_envs": NUM_ENVS,
                "profile": "omni_full_episode_v1",
                "seed": 42,
                "checkpoint": str(checkpoint),
                "reference": str(reference),
                "coefficient_donor": str(coefficient_donor),
                "checkpoint_sha256": checkpoint_hash,
                "reference_sha256": reference_hash,
                "coefficient_donor_sha256": coefficient_donor_hash,
                "checkpoint_artifacts_written": [],
                "budget": {"control_steps": total_steps, "simulated_seconds": summary["simulated_seconds"]},
                "state_hashes_before": before,
                "state_hashes_after": after,
                "forbidden_calls": [],
                "storage_cleared_each_horizon": True,
                "stats_environment_patch_restored": True,
                "mutation_patches_restored": True,
            })
        except BaseException as exc:
            if stats is not None:
                stats.close()
            write_json(status_path, {
                "state": "AUDIT_FAILED", "authoritative": True, "smoke_only": smoke,
                "completed_updates": EXPECTED_COMPLETED_UPDATES, "optimizer_updates": 0,
                "error": str(exc), "traceback": traceback.format_exc(), "state_hashes_before": before,
                "forbidden_calls": [] if guard is None else guard.calls,
            })
            raise

    return audit_learn


def main() -> None:
    args = parse_args()
    root, code, output = args.root.resolve(), args.code.resolve(), args.output.resolve()
    checkpoint, reference = args.checkpoint.resolve(), args.reference.resolve()
    coefficient_donor = args.coefficient_donor.resolve()
    if output.exists():
        raise FileExistsError(f"Output must be new: {output}")
    for path in (code / "train.py", checkpoint, reference, coefficient_donor):
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path[:0] = [str(code), str(root / "upstream/source/rsl_rl")]
    import torch
    from rsl_rl.runners import OnPolicyRunnerCTS

    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reference_state = torch.load(reference, map_location="cpu", weights_only=False)
    donor_state = torch.load(coefficient_donor, map_location="cpu", weights_only=False)
    if int(checkpoint_state["completed_updates"]) != EXPECTED_COMPLETED_UPDATES:
        raise ValueError("Checkpoint is not V9 completed_updates=44711")
    if int(reference_state["completed_updates"]) != EXPECTED_REFERENCE_COMPLETED_UPDATES:
        raise ValueError("Reference is not V6 completed_updates=43202")
    if int(donor_state["completed_updates"]) != EXPECTED_DONOR_COMPLETED_UPDATES:
        raise ValueError("Coefficient donor is not V11 completed_updates=44759")
    del checkpoint_state, reference_state, donor_state
    checkpoint_hash, reference_hash = file_sha256(checkpoint), file_sha256(reference)
    coefficient_donor_hash = file_sha256(coefficient_donor)
    stats_path = Path(__file__).with_name("audit_stats.py")
    original_learn, original_argv = OnPolicyRunnerCTS.learn, sys.argv[:]
    OnPolicyRunnerCTS.learn = make_audit_learn(
        torch, stats_path, output, args.smoke, checkpoint, reference, coefficient_donor,
        checkpoint_hash, reference_hash, coefficient_donor_hash
    )
    sys.argv = [
        str(code / "train.py"), "--root", str(root), "--output", str(output),
        "--num-envs", str(NUM_ENVS), "--updates", "0", "--mixed-pit",
        "--resume", str(checkpoint), "--startup-repair-profile", "omni_full_episode_v1",
        "--startup-anchor-reference", str(reference),
        "--startup-anchor-coefficients-from", str(coefficient_donor),
    ]
    try:
        runpy.run_path(str(code / "train.py"), run_name="__main__")
    finally:
        sys.argv = original_argv
        OnPolicyRunnerCTS.learn = original_learn
    status = json.loads((output / "audit_status.json").read_text(encoding="utf-8"))
    expected = "AUDIT_SMOKE_COMPLETE" if args.smoke else "AUDIT_COMPLETE"
    if status.get("state") != expected:
        raise RuntimeError(status)
    status.update({
        "checkpoint": str(checkpoint), "reference": str(reference),
        "coefficient_donor": str(coefficient_donor),
        "checkpoint_sha256": checkpoint_hash, "reference_sha256": reference_hash,
        "coefficient_donor_sha256": coefficient_donor_hash,
        "input_files_unchanged": (
            file_sha256(checkpoint) == checkpoint_hash
            and file_sha256(reference) == reference_hash
            and file_sha256(coefficient_donor) == coefficient_donor_hash
        ),
        "checkpoint_artifacts_written": [p.name for p in output.glob("model_*.pt")],
        "launcher_patch_restored": OnPolicyRunnerCTS.learn is original_learn,
        "training_success_receipt_superseded": True,
    })
    write_json(output / "audit_status.json", status)
    print(json.dumps({"state": "audit_requires_external_finalization", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
