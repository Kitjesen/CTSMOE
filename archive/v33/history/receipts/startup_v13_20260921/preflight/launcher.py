"""Run the V13 60-second zero-update preflight through the real train path."""

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
PROFILE = "omni_full_episode_v13"


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
        try:
            self._forbid(self.runner.alg, "update", "alg.update")
            self._forbid(self.runner.alg, "compute_returns", "alg.compute_returns")
            self._forbid(self.runner.alg.optimizer, "step", "alg.optimizer.step")
            self._forbid(self.runner.alg.optimizer_stu_enc, "step", "alg.optimizer_stu_enc.step")
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
    spec = importlib.util.spec_from_file_location("startup_v13_audit_stats", path)
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
    inputs: dict[str, Path],
    hashes: dict[str, str],
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
            if init_at_random_ep_len or self.alg.storage.step != 0:
                raise RuntimeError("Preflight requires real episode age zero and empty rollout storage")
            raw = self.env.unwrapped
            policy = self.alg.policy
            before = {
                "model": state_sha256(policy.state_dict(), torch),
                "ppo_optimizer": state_sha256(self.alg.optimizer.state_dict(), torch),
                "student_optimizer": state_sha256(self.alg.optimizer_stu_enc.state_dict(), torch),
            }
            stats = load_stats(stats_path)(raw, self.alg, output)
            obs = self.env.get_observations().to(self.device)
            self.train_mode()
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
            write_json(status_path, {
                "state": "AUDIT_SMOKE_COMPLETE" if smoke else "AUDIT_COMPLETE",
                "authoritative": True,
                "written_before_learn_return": True,
                "smoke_only": smoke,
                "completed_updates": EXPECTED_COMPLETED_UPDATES,
                "optimizer_updates": 0,
                "control_steps": total_steps,
                "simulated_seconds": summary["simulated_seconds"],
                "num_envs": NUM_ENVS,
                "profile": PROFILE,
                "seed": 42,
                **{name: str(path) for name, path in inputs.items()},
                **{f"{name}_sha256": value for name, value in hashes.items()},
                "checkpoint_artifacts_written": [],
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
    if output.exists():
        raise FileExistsError(f"Output must be new: {output}")
    inputs = {
        "checkpoint": args.checkpoint.resolve(),
        "reference": args.reference.resolve(),
        "coefficient_donor": args.coefficient_donor.resolve(),
    }
    for path in (code / "train.py", *inputs.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path[:0] = [str(code), str(root / "upstream/source/rsl_rl")]
    import torch
    from rsl_rl.runners import OnPolicyRunnerCTS

    expected_updates = {
        "checkpoint": EXPECTED_COMPLETED_UPDATES,
        "reference": EXPECTED_REFERENCE_COMPLETED_UPDATES,
        "coefficient_donor": EXPECTED_DONOR_COMPLETED_UPDATES,
    }
    for name, path in inputs.items():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if int(state["completed_updates"]) != expected_updates[name]:
            raise ValueError(f"Unexpected {name} completed_updates")
        del state
    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    stats_path = Path(__file__).with_name("audit_stats.py")
    original_learn, original_argv = OnPolicyRunnerCTS.learn, sys.argv[:]
    OnPolicyRunnerCTS.learn = make_audit_learn(torch, stats_path, output, args.smoke, inputs, hashes)
    sys.argv = [
        str(code / "train.py"), "--root", str(root), "--output", str(output),
        "--num-envs", str(NUM_ENVS), "--updates", "0", "--mixed-pit",
        "--resume", str(inputs["checkpoint"]), "--startup-repair-profile", PROFILE,
        "--startup-anchor-reference", str(inputs["reference"]),
        "--startup-anchor-coefficients-from", str(inputs["coefficient_donor"]),
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
        "input_files_unchanged": all(file_sha256(inputs[name]) == value for name, value in hashes.items()),
        "checkpoint_artifacts_written": [path.name for path in output.glob("model_*.pt")],
        "launcher_patch_restored": OnPolicyRunnerCTS.learn is original_learn,
        "training_success_receipt_superseded": True,
    })
    write_json(output / "audit_status.json", status)
    print(json.dumps({"state": "audit_requires_external_finalization", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
