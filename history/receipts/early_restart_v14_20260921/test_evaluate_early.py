import json
from pathlib import Path

import pytest

from early_protocol import (FIXED_RUNTIME, PROTOCOL_ID, build_cases,
                            checkpoint_metadata, load_protocol, load_selection)


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def valid_protocol():
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "purpose": "Common physics protocol, not a historical training reconstruction.",
        "runtime": dict(FIXED_RUNTIME),
        "sources": [
            {"path": "runs/a/model_29999.pt", "completed_updates": 30000,
             },
            {"path": "runs/b/model_31830.pt", "completed_updates": 31831,
             },
        ],
    }


def test_protocol_and_selection_require_explicit_metadata(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol = valid_protocol()
    write_json(protocol_path, protocol)
    loaded = load_protocol(protocol_path)
    selection = [
        {"path": source["path"], "label": f"p{i}_{mode}", "teacher": teacher}
        for i, source in enumerate(protocol["sources"])
        for mode, teacher in (("student", False), ("teacher", True))
    ]
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)
    assert load_selection(selection_path, loaded) == selection

    del protocol["sources"][0]["completed_updates"]
    write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="completed_updates"):
        load_protocol(protocol_path)


def test_runtime_recipe_is_exact_and_wave_cannot_be_disabled(tmp_path):
    path = tmp_path / "protocol.json"
    protocol = valid_protocol()
    protocol["runtime"]["wave_reset_profile"] = "original"
    write_json(path, protocol)
    with pytest.raises(ValueError, match="wave_reset_profile"):
        load_protocol(path)


def test_selection_requires_student_and_teacher_for_every_source(tmp_path):
    protocol_path, selection_path = tmp_path / "p.json", tmp_path / "s.json"
    write_json(protocol_path, valid_protocol())
    selection = [
        {"path": "runs/a/model_29999.pt", "label": "a", "teacher": False},
        {"path": "runs/a/model_29999.pt", "label": "at", "teacher": True},
        {"path": "runs/b/model_31830.pt", "label": "b", "teacher": False},
    ]
    write_json(selection_path, selection)
    with pytest.raises(ValueError, match="student and one teacher"):
        load_selection(selection_path, load_protocol(protocol_path))


def test_case_counts_and_default_pose_contract():
    selection = [
        {"path": path, "label": label, "teacher": teacher}
        for path, label in (("a", "30000"), ("b", "31831"))
        for teacher in (False, True)
    ]
    terrain = build_cases(selection, "terrain83")
    flat25 = build_cases(selection, "flat25")
    assert len(terrain) == 332
    assert len(flat25) == 80
    assert all(case["mode"] in {"direct", "settle"} for case in terrain + flat25)
    assert all(case["terrain"] == "flat" for case in flat25)
    assert {case["terrain"] for case in terrain} == {
        "flat", "stairs_up", "stairs_down", "slope_up", "slope_down",
        "wave", "rough_slope", "obstacles", "pit",
    }


def test_checkpoint_metadata_does_not_fallback_to_checkpoint_profiles():
    contract = {"completed_updates": 30000}
    state = {"model_state_dict": {}, "startup_repair_profile": "unknown-old-profile"}
    assert checkpoint_metadata(state, contract)["completed_updates"] == 30000
    with pytest.raises(ValueError, match="model_state_dict"):
        checkpoint_metadata({}, contract)
    with pytest.raises(ValueError, match="completed_updates"):
        checkpoint_metadata({"model_state_dict": {}, "completed_updates": 3}, contract)


def test_evaluator_contains_no_training_step():
    source = Path(__file__).with_name("evaluate_early.py").read_text(encoding="utf-8")
    assert "optimizer.step" not in source
    assert "runner.learn" not in source
    assert "training_updates=0" in source
