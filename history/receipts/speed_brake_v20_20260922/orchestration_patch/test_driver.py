import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "d", Path(__file__).with_name("driver.py")
)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def test_fixed_targets_and_budget():
    assert d.TARGETS == (34823, 37823, 41823)
    assert d.budget(41823) == 10000
    try:
        d.budget(41824)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_first_increment_is_target_minus_source():
    assert d.TARGETS[0] - 33100 == 1723
