import ast
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "reinforcement_learning"
    / "rsl_rl"
    / "evaluate_torque_tracking.py"
)


def _load_helpers():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = {"_command_scale", "_difference_stats"}
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    exec(compile(module, SCRIPT, "exec"), namespace)
    return namespace


def test_command_profiles_start_and_converge():
    scale = _load_helpers()["_command_scale"]

    assert scale("fixed", 0, 10, 5) == 1.0
    assert scale("step", 9, 10, 5) == 0.0
    assert scale("step", 10, 10, 5) == 1.0
    assert scale("ramp", 10, 10, 5) == 0.0
    assert scale("ramp", 12, 10, 5) == 0.5
    assert scale("ramp", 14, 10, 5) == 1.0
    assert scale("ramp", 20, 10, 5) == 1.0


def test_action_difference_statistics():
    stats = _load_helpers()["_difference_stats"]
    values = torch.tensor([[[0.0]], [[1.0]], [[3.0]], [[6.0]]])

    result = stats(values)

    assert result["first_difference"] == {"rms": 2.160246849, "max_abs": 3.0}
    assert result["second_difference"] == {"rms": 1.0, "max_abs": 1.0}


def test_top_level_gate_is_explicitly_recovery_only():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"recovery_passed": all(' in source
    assert '"passed": all(' not in source
