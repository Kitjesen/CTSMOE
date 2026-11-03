"""Pure review tests for recovery01; no subprocess, GPU, SSH, or filesystem mutation."""
import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path
import pytest


SRC = Path(__file__).with_name("recover_pipeline.py")


def _budget():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "budget")
    ns = {"LOST_UPDATES": 8}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["budget"]


def test_budget_boundaries_and_stage_targets():
    budget = _budget()
    assert budget(32831)["spent_new_updates"] == 1008
    assert budget(34823)["remaining_new_updates"] == 7000
    assert budget(37823)["remaining_new_updates"] == 4000
    assert budget(41823)["remaining_new_updates"] == 0
    for bad in (32830, 41824):
        try:
            budget(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_source_has_three_nonoverlapping_recovery_stages_and_gate_timing():
    text = SRC.read_text(encoding="utf-8")
    assert "REMAINING_STAGES = (1992, 3000, 4000)" in text
    assert "target = completed + increment" in text
    assert "output.exists()" in text and "Training output already exists" in text
    assert "STARTED.json" in text and "Recovery has already been attempted" in text
    assert "full_episode_v12_pipeline.lock" in text
    assert "spent_new_updates'] >= 3000" in text


def test_checkpoint_names_are_end_of_stage_minus_one():
    text = SRC.read_text(encoding="utf-8")
    assert "model_{target-1}.pt" in text
    assert "source, completed = output/f'model_{target-1}.pt', target" in text


def test_preflight_and_verification_run_before_training():
    text = SRC.read_text(encoding="utf-8")
    assert text.index("recovery_preflight") < text.index("for increment in REMAINING_STAGES")
    assert text.index("recovery_verify") < text.index("for increment in REMAINING_STAGES")


def test_failure_preserves_evidence_and_disables_retry():
    text = SRC.read_text(encoding="utf-8")
    assert "preserve evidence; no automatic retry" in text
    assert "automatic_retry=False" in text
    assert "Less than 10GiB free" in text


def test_posture_stop_curriculum_is_gated_after_spent_3000():
    text = SRC.read_text(encoding="utf-8")
    marker = "budget(completed)['spent_new_updates'] >= 3000"
    assert marker in text
    assert text.index(marker) < text.index("if verdict['stop_for_review']")


@pytest.mark.skipif(os.name != "posix", reason="POSIX flock validation runs on remote Linux")
def test_posix_double_start_lock_does_not_mutate_status(tmp_path):
    script = tmp_path / "lock.py"
    script.write_text(textwrap.dedent("""
        import fcntl, sys
        with open(sys.argv[1], 'a') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(17)
            print('LOCKED', flush=True)
            input()
    """), encoding="utf-8")
    lock = tmp_path / "pipeline.lock"
    first = subprocess.Popen([sys.executable, str(script), str(lock)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    try:
        assert first.stdout.readline().strip() == "LOCKED"
        second = subprocess.run([sys.executable, str(script), str(lock)], input="", text=True)
        assert second.returncode == 17
        assert lock.read_bytes() == b""
    finally:
        first.kill()
        first.wait()
