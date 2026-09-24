"""Structural state fingerprints and forbidden calls during zero-update rollout."""

from __future__ import annotations

import hashlib
import math


def state_sha256(value, torch) -> str:
    result = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite state tensor")
            result.update(str(tensor.dtype).encode())
            result.update(repr(tuple(tensor.shape)).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            result.update(str(len(item)).encode())
            for value in item:
                visit(value)
        elif item is None or isinstance(item, (str, int, float, bool)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Nonfinite state scalar")
            result.update(type(item).__name__.encode())
            result.update(repr(item).encode())
        else:
            raise TypeError(f"Unsupported state member: {type(item)}")

    visit(value)
    return result.hexdigest()


class MutationGuard:
    def __init__(self, runner):
        self.runner = runner
        self.calls = []
        self.originals = []
        self.restored = False

    def __enter__(self):
        for owner, name, label in (
            (self.runner.alg, "update", "alg.update"),
            (self.runner.alg, "compute_returns", "alg.compute_returns"),
            (self.runner.alg.optimizer, "step", "ppo_optimizer.step"),
            (self.runner.alg.optimizer_stu_enc, "step", "student_optimizer.step"),
            (self.runner, "save", "runner.save"),
        ):
            original = getattr(owner, name)

            def forbidden(*_args, _label=label, **_kwargs):
                self.calls.append(_label)
                raise RuntimeError(f"Forbidden zero-update call: {_label}")

            self.originals.append((owner, name, original))
            setattr(owner, name, forbidden)
        return self

    def __exit__(self, *_args):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.restored = all(getattr(owner, name) is original
                            for owner, name, original in self.originals)
