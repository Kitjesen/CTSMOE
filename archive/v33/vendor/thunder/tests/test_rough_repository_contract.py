"""Repository-level checks for the Thunder rough experiment bundle."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUGH_SCRIPTS = PROJECT_ROOT / "scripts" / "experiments" / "rough"


def test_rough_experiment_bundle_has_no_host_specific_paths():
    expected = {
        "run_from_zero_v0.sh",
        "run_command_sampling_ab_from_13800.sh",
        "run_reward_c_from_13800.sh",
        "queue_reward_c_from_13800.sh",
        "evaluate_checkpoint.sh",
    }
    assert {path.name for path in ROUGH_SCRIPTS.glob("*.sh")} == expected
    for path in ROUGH_SCRIPTS.glob("*.sh"):
        source = path.read_text(encoding="utf-8")
        assert "/root/autodl-tmp/" not in source
        assert "DOSO_ROOT" in source
        assert source.startswith("#!/usr/bin/env bash\n")


def test_reward_c_locks_parent_lineage_and_fixed_evaluation():
    launcher = (ROUGH_SCRIPTS / "run_reward_c_from_13800.sh").read_text(
        encoding="utf-8"
    )
    queue = (ROUGH_SCRIPTS / "queue_reward_c_from_13800.sh").read_text(encoding="utf-8")
    assert "EXPECTED_SOURCE_SHA256" in launcher
    assert (
        "b79c66749f5e1147e1c780c0ba7a35790c212b28f8789012e6fb97c8097e0727" in launcher
    )
    assert "env.rewards.feet_stumble.weight=-1.0" in launcher
    assert "env.rewards.undesired_contacts.weight=0.0" in launcher
    assert "env.rewards.contact_forces.weight=0.0" in launcher
    assert "env.rewards.touchdown_velocity_l2.weight=0.0" in launcher
    assert 'RUN_FIXED_EVAL="${RUN_FIXED_EVAL:-1}"' in queue
    assert 'bash "$EVALUATOR"' in queue


def test_evaluator_uses_frozen_v0_configuration():
    evaluator = (
        PROJECT_ROOT
        / "scripts"
        / "reinforcement_learning"
        / "rsl_rl"
        / "evaluate_rough.py"
    ).read_text(encoding="utf-8")
    assert "DosoThunderTerrainInwardRoughV0EnvCfg" in evaluator
    assert '"randomize_foot_material_with_privileged"' in evaluator
    assert "weights_only=True" in evaluator
    assert "CONTACT_FORCE_HISTORY_LENGTH = 3" in evaluator
    assert 'raise RuntimeError("Contact force history is unavailable")' in evaluator


def test_evaluator_archives_checkpoint_lineage_and_raw_contact_schema():
    evaluator_path = (
        PROJECT_ROOT
        / "scripts"
        / "reinforcement_learning"
        / "rsl_rl"
        / "evaluate_rough.py"
    )
    tree = ast.parse(evaluator_path.read_text(encoding="utf-8"))
    string_keys = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {
        "checkpoint_bytes",
        "checkpoint_sha256",
        "contact_rate_denominator",
        "foot_force_sampling",
        "foot_force_history_length",
        "non_foot_contact_rate",
        "foot_force_p95_n",
        "foot_force_p99_n",
        "foot_force_peak_n",
        "touchdown_count",
        "touchdown_downward_speed_p95_mps",
        "touchdown_downward_speed_p99_mps",
        "touchdown_downward_speed_peak_mps",
        "stumble_rate",
    } <= string_keys

    queue = (ROUGH_SCRIPTS / "queue_reward_c_from_13800.sh").read_text(encoding="utf-8")
    assert "full_checkpoint_bytes=" in queue
    assert "full_checkpoint_sha256=" in queue
    assert "checkpoint_sha256=$full_checkpoint_sha256" in queue
