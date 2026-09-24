"""Pure protocol validation and case construction for early zero-update evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PROTOCOL_ID = "early_restart_v14_zero_update_v1"
FIXED_RUNTIME = {
    "actuator_profile": "rs0402_transient_v4",
    "slope_profile": "slope45_v1",
    "stair_command_profile": "straight_speed_v1",
    "wave_reset_profile": "wave_surface_v1",
    "pit_training_profile": "strict_v1",
    "base_height_penalty": "off",
    "pit_posture_profile": "off",
    "surface_posture_profile": "off",
    "stance_strength_profile": "off",
    "startup_repair_profile": "off",
    "command_curriculum_start": "intermediate",
    "common_step_counter": 0,
}
COMMANDS = (
    ("forward", (.6, 0., 0.)),
    ("reverse", (-.6, 0., 0.)),
    ("left", (0., .4, 0.)),
    ("right", (0., -.4, 0.)),
    ("yaw_left", (0., 0., .6)),
    ("yaw_right", (0., 0., -.6)),
    ("diagonal", (.4, .3, 0.)),
    ("diagonal_back", (-.4, -.3, 0.)),
    ("forward_turn", (.4, 0., .4)),
    ("side_turn", (0., .3, -.4)),
)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    """Load an explicit protocol; never infer missing checkpoint metadata."""
    protocol = _object(json.loads(path.read_text(encoding="utf-8")), "protocol")
    required = {"schema_version", "protocol_id", "purpose", "runtime", "sources"}
    missing = sorted(required - protocol.keys())
    if missing:
        raise ValueError(f"protocol missing required fields: {missing}")
    if protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {protocol['schema_version']!r}")
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise ValueError(f"unexpected protocol_id: {protocol['protocol_id']!r}")
    if not isinstance(protocol["purpose"], str) or not protocol["purpose"].strip():
        raise ValueError("protocol purpose must be a non-empty string")
    runtime = _object(protocol["runtime"], "runtime")
    if runtime != FIXED_RUNTIME:
        missing_runtime = sorted(set(FIXED_RUNTIME) - set(runtime))
        extra_runtime = sorted(set(runtime) - set(FIXED_RUNTIME))
        wrong = {k: (runtime.get(k), v) for k, v in FIXED_RUNTIME.items()
                 if k in runtime and runtime[k] != v}
        raise ValueError(
            f"runtime must exactly match the fixed evaluation recipe; "
            f"missing={missing_runtime}, extra={extra_runtime}, wrong={wrong}"
        )
    sources = protocol["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    seen: set[str] = set()
    for index, source in enumerate(sources):
        source = _object(source, f"sources[{index}]")
        required_source = {"path", "completed_updates"}
        absent = sorted(required_source - source.keys())
        if absent:
            raise ValueError(f"sources[{index}] missing required fields: {absent}")
        if set(source) != required_source:
            raise ValueError(f"sources[{index}] has unsupported fields: {sorted(set(source)-required_source)}")
        if not isinstance(source["path"], str) or not source["path"]:
            raise ValueError(f"sources[{index}].path must be a non-empty string")
        if source["path"] in seen:
            raise ValueError(f"duplicate source path: {source['path']}")
        seen.add(source["path"])
        for field in ("completed_updates",):
            if not isinstance(source[field], int) or isinstance(source[field], bool) or source[field] < 0:
                raise ValueError(f"sources[{index}].{field} must be a non-negative integer")
    return protocol


def load_selection(path: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate externally supplied policy rows against explicit source metadata."""
    selection = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(selection, list) or not selection:
        raise ValueError("selection must be a non-empty list")
    source_paths = {source["path"] for source in protocol["sources"]}
    pairs: dict[str, set[bool]] = {path: set() for path in source_paths}
    for index, item in enumerate(selection):
        item = _object(item, f"selection[{index}]")
        if set(item) != {"path", "label", "teacher"}:
            raise ValueError(
                f"selection[{index}] must contain exactly path, label, teacher; got {sorted(item)}"
            )
        if item["path"] not in source_paths:
            raise ValueError(f"selection[{index}] path has no explicit source contract: {item['path']!r}")
        if not isinstance(item["label"], str) or not item["label"]:
            raise ValueError(f"selection[{index}].label must be a non-empty string")
        if type(item["teacher"]) is not bool:
            raise ValueError(f"selection[{index}].teacher must be boolean")
        if item["teacher"] in pairs[item["path"]]:
            raise ValueError(f"duplicate student/teacher row for {item['path']!r}")
        pairs[item["path"]].add(item["teacher"])
    missing = {path: sorted({False, True} - modes) for path, modes in pairs.items() if modes != {False, True}}
    if missing:
        raise ValueError(f"every protocol source requires one student and one teacher row: {missing}")
    if len(selection) != 2 * len(source_paths):
        raise ValueError("selection must cover every source exactly once in student and teacher mode")
    return selection


def source_contracts(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {source["path"]: source for source in protocol["sources"]}


def build_cases(selection: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    if suite not in {"terrain83", "flat25"}:
        raise ValueError(f"unknown suite: {suite!r}")
    cases: list[dict[str, Any]] = []
    for policy, item in enumerate(selection):
        for mode in ("direct", "settle"):
            for direction, velocity in COMMANDS:
                cases.append(dict(policy=policy, terrain="flat", level=0, mode=mode,
                                  direction=direction, velocity=list(velocity),
                                  teacher=item["teacher"], suite=suite))
        if suite == "terrain83":
            for terrain in ("stairs_up", "stairs_down", "slope_up", "slope_down",
                            "wave", "rough_slope", "obstacles"):
                for level in (6, 9):
                    for speed in (.3, .6, 1.2):
                        cases.append(dict(policy=policy, terrain=terrain, level=level,
                                          mode="direct", direction="forward",
                                          velocity=[speed, 0., 0.], teacher=item["teacher"],
                                          suite=suite))
            for level in (0, 2, 4, 6, 9, 15, 21):
                for speed in (.3, .6, .9):
                    cases.append(dict(policy=policy, terrain="pit", level=level,
                                      mode="direct", direction="forward",
                                      velocity=[speed, 0., 0.], teacher=item["teacher"],
                                      suite=suite))
    per_policy = 83 if suite == "terrain83" else 20
    if len(cases) != per_policy * len(selection):
        raise AssertionError("case construction count mismatch")
    cases.sort(key=lambda case: case["policy"])
    for case_id, case in enumerate(cases):
        case["id"] = case_id
    return cases


def checkpoint_metadata(state: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Validate identity only; runtime behavior always comes from the protocol recipe."""
    if "model_state_dict" not in state:
        raise ValueError("checkpoint missing model_state_dict")
    if "completed_updates" in state and state["completed_updates"] != contract["completed_updates"]:
        raise ValueError("checkpoint completed_updates disagrees with explicit source contract")
    return {
        "completed_updates": contract["completed_updates"],
        "checkpoint_declared_completed_updates": state.get("completed_updates"),
        "checkpoint_declared_common_step_counter": state.get("common_step_counter"),
    }
