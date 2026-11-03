import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("driver", Path(__file__).with_name("continue.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_budget_and_checkpoints():
    assert sum(driver.STAGES) == driver.TOTAL == 9700
    assert driver.planned_completions() == [32831, 34831, 36831, 38831, 41831]
    assert 41831 - 31831 == 10000


def test_collapse_is_consecutive_and_role_specific():
    s = driver.update_collapse_streak({}, {"student": 9, "teacher": 20})
    assert s == {"student": 1, "teacher": 0}
    s = driver.update_collapse_streak(s, {"student": 20, "teacher": 9})
    assert s == {"student": 0, "teacher": 1}
    s = driver.update_collapse_streak(s, {"student": 20, "teacher": 9})
    assert s == {"student": 0, "teacher": 2}
    assert max(driver.update_collapse_streak(s, {"student": 10, "teacher": 10}).values()) == 0


def test_missing_policy_or_evaluation_cannot_look_successful():
    with pytest.raises(ValueError):
        driver.flat_counts([], Path("candidate.pt"))
    with pytest.raises(ValueError):
        driver.update_collapse_streak({}, {"student": 20})
    with pytest.raises(ValueError):
        driver.validate_evaluation({}, {"cases": []}, 80, Path("candidate.pt"), 32831)
