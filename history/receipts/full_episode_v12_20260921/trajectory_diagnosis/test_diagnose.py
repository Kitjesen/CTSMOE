import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("diagnose.py")
SPEC = importlib.util.spec_from_file_location("trajectory_diagnose", MODULE_PATH)
assert SPEC and SPEC.loader
DIAGNOSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSE)


def test_window_bounds_clamps_to_trace() -> None:
    assert DIAGNOSE.window_bounds(2.0, 3.0, 0.02, 125) == (100, 125)


def test_peak_reports_sample_time_and_column() -> None:
    values = np.array([[0.0, -2.0], [3.0, 1.0]])
    assert DIAGNOSE.peak(values, 0.02) == {"value": 3.0, "time_s": 0.04, "column": 0}


def test_case_key_ignores_policy_identity() -> None:
    case = {"terrain": "flat", "level": 0, "mode": "direct", "direction": "forward", "velocity": [0.6, 0, 0]}
    changed = {**case, "policy": 3, "label": "candidate"}
    assert DIAGNOSE.case_key(case) == DIAGNOSE.case_key(changed)
