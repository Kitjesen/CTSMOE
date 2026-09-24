"""Static and local contract checks for the Isaac audit launcher."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace


LAUNCHER = Path(__file__).with_name("launcher.py")
SOURCE = LAUNCHER.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def load_launcher():
    spec = importlib.util.spec_from_file_location("distribution_audit_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_entry_is_fail_closed_and_zero_update() -> None:
    required = (
        '"--updates",\n        "0"',
        '"--num-envs",\n        str(NUM_ENVS)',
        '"--mixed-pit"',
        '"--resume"',
        '"--startup-anchor-reference"',
        'self._forbid(alg, "update", "alg.update")',
        'self._forbid(alg, "compute_returns", "alg.compute_returns")',
        'self._forbid(alg.optimizer, "step", "alg.optimizer.step")',
        'self._forbid(alg.optimizer_stu_enc, "step", "alg.optimizer_stu_enc.step")',
        'self._forbid(self.runner, "save", "runner.save")',
    )
    for text in required:
        assert text in SOURCE


def test_native_rollout_order_and_stats_boundary() -> None:
    function = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "audit_learn"
    )
    calls = sorted(
        ((node.lineno, ast.unparse(node.func)) for node in ast.walk(function) if isinstance(node, ast.Call)),
        key=lambda item: item[0],
    )
    positions = {
        name: next(line for line, called in calls if called == name)
        for name in (
            "self.alg.act",
            "stats.before",
            "self.env.step",
            "self.alg.process_env_step",
            "stats.after",
            "self.alg.storage.clear",
            "stats.finish",
        )
    }
    assert list(positions.values()) == sorted(positions.values())


def test_budget_and_nonmutation_receipt_are_explicit() -> None:
    assert "HORIZONS = 20" in SOURCE
    assert "STEPS_PER_HORIZON = 24" in SOURCE
    assert "SMOKE_STEPS = 3" in SOURCE
    assert '"optimizer_updates": 0' in SOURCE
    assert 'checkpoint_artifacts_written=[]' in SOURCE
    assert "before_hashes != after_hashes" in SOURCE
    assert 'output.exists()' in SOURCE


def test_mutation_guard_fails_closed_and_restores_every_entry_point() -> None:
    launcher = load_launcher()
    counts = {name: 0 for name in ("update", "returns", "ppo", "student", "save")}

    def mark(name):
        def call(*_args, **_kwargs):
            counts[name] += 1

        return call

    ppo = SimpleNamespace(step=mark("ppo"))
    student = SimpleNamespace(step=mark("student"))
    alg = SimpleNamespace(
        update=mark("update"),
        compute_returns=mark("returns"),
        optimizer=ppo,
        optimizer_stu_enc=student,
    )
    runner = SimpleNamespace(alg=alg, save=mark("save"))
    guard = launcher.MutationGuard(runner)
    with guard:
        for owner, attribute in (
            (alg, "update"),
            (alg, "compute_returns"),
            (ppo, "step"),
            (student, "step"),
            (runner, "save"),
        ):
            try:
                getattr(owner, attribute)()
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"{attribute} was not forbidden")
    assert guard.restored
    assert len(guard.calls) == 5
    alg.update()
    alg.compute_returns()
    ppo.step()
    student.step()
    runner.save()
    assert counts == {"update": 1, "returns": 1, "ppo": 1, "student": 1, "save": 1}


def test_state_hash_supports_scalar_optimizer_tensors() -> None:
    import torch

    launcher = load_launcher()
    state = {
        "state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.arange(4.0)}},
        "param_groups": [{"lr": 0.001, "params": [0]}],
    }
    first = launcher.state_sha256(state, torch)
    second = launcher.state_sha256(state, torch)
    assert first == second
    state["state"][0]["step"] += 1
    assert launcher.state_sha256(state, torch) != first
