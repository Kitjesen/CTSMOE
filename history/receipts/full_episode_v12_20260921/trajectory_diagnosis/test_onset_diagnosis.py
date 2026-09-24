import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).with_name("onset_diagnosis.py")
SPEC = importlib.util.spec_from_file_location("onset_diagnosis", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_first_crossing_uses_time_then_column_order() -> None:
    values = np.array([[0.0, 0.7], [0.8, 0.0]])
    assert MODULE.first_crossing(values, 0.6, 0.02) == {
        "time_s": 0.02,
        "column": 1,
        "value": 0.7,
    }


def test_peak_preserves_signed_value() -> None:
    values = np.array([[0.1, -0.9], [0.8, 0.0]])
    assert MODULE.peak(values, 0.02) == {"time_s": 0.02, "column": 1, "value": -0.9}
