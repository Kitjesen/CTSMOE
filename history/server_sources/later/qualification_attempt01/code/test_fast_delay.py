"""Differential tests against the exact server Isaac Lab buffer implementation."""
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch

from fast_delay import FastDelayBuffer


def reference_type():
    package = "_isaac_delay_reference"
    folder = Path(os.environ.get("ISAAC_BUFFER_SOURCE", Path(__file__).parent / "reference"))
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(folder)]
        sys.modules[package] = module
        for name in ("circular_buffer", "delay_buffer"):
            spec = importlib.util.spec_from_file_location(f"{package}.{name}", folder / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return sys.modules[f"{package}.delay_buffer"].DelayBuffer


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("horizon", [0, 1, 4])
@pytest.mark.parametrize("shape", [(17, 3), (17, 2, 3)])
def test_compute_history_and_reset_parity(device, horizon, shape):
    cls = reference_type()
    baseline = cls(horizon, shape[0], device)
    fast = FastDelayBuffer(cls(horizon, shape[0], device))
    lags = torch.arange(shape[0], device=device, dtype=torch.int) % (horizon + 1)
    baseline.set_time_lag(lags)
    fast.set_time_lag(lags)
    for step in range(65):
        if step in (0, 11, 35):
            for ids in (torch.tensor([1, 5, 9], device=device), [3, 5], []):
                baseline.reset(ids)
                fast.reset(ids)
                if isinstance(ids, torch.Tensor):
                    ids.fill_(7)  # pending-reset IDs must not alias caller memory
        if step == 17:
            baseline.reset(slice(None))
            fast.reset(slice(None))
        if step == 23:
            baseline.reset()
            fast.reset()
        if step % 7 == 0:
            new_lags = torch.flip(lags, [0])
            baseline.set_time_lag(new_lags)
            fast.set_time_lag(new_lags)
        if step % 9 == 0:
            ids = torch.tensor([2, 6, 10], device=device)
            subset_lags = torch.tensor([0, horizon, horizon // 2], device=device, dtype=torch.int)
            baseline.set_time_lag(subset_lags, ids)
            fast.set_time_lag(subset_lags, ids)
        data = (torch.arange(torch.tensor(shape).prod().item(), device=device).reshape(shape) + 1000 * step).float()
        expected = baseline.compute(data)
        actual = fast.compute(data)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(fast._ring.buffer, baseline._circular_buffer.buffer, rtol=0, atol=0)
        actual.fill_(-999)  # returned result must be independent of stored history
        torch.testing.assert_close(fast._ring.buffer, baseline._circular_buffer.buffer, rtol=0, atol=0)


def test_errors_and_no_random_calls():
    cls = reference_type()
    fast = FastDelayBuffer(cls(4, 3, "cpu"))
    before = torch.get_rng_state()
    fast.compute(torch.ones(3, 2))
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    with pytest.raises(ValueError):
        fast.compute(torch.ones(2, 2))
    with pytest.raises(ValueError):
        FastDelayBuffer(fast._reference)
    for lag in [-1, 5]:
        with pytest.raises(ValueError):
            FastDelayBuffer(cls(4, 3, "cpu")).set_time_lag(lag)
    with pytest.raises(TypeError):
        FastDelayBuffer(cls(4, 3, "cpu")).set_time_lag(torch.ones(3))
