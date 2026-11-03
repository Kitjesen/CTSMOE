"""CPU-only independent fixed-case gate; record source L0 failure honestly."""

from __future__ import annotations

from collections import Counter
import json
import re

from contract import (HERE, SOURCE, SOURCE_COMPLETED, SOURCE_SHA256,
                      V21_FIXED_PROTOCOL, V23_ASSESSMENT, digest, load_json,
                      verify_frozen, write_new)


ROUTE = HERE / "source_route_fixed"
OLD = HERE / "source_old_nine_fixed"
SUITES = {"terrain83": 166, "flat25": 40,
          "stairend": 48, "stairend_15cm": 48}
OLD_TYPES = {"flat", "slope_up", "slope_down", "rough_slope",
             "stairs_up", "stairs_down", "wave", "obstacles", "pit"}


def verify_route(manifest, arrays) -> dict:
    import numpy as np

    cases = manifest.get("cases", [])
    if (manifest.get("state") != "complete" or manifest.get("source") != str(SOURCE)
            or manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("source_completed_updates") != SOURCE_COMPLETED
            or manifest.get("optimizer_updates") != 0
            or manifest.get("route_level") != 0
            or manifest.get("fixed_cases") != 12 or len(cases) != 12
            or manifest.get("speeds_mps") != [0.2, 0.6, 1.0]
            or manifest.get("seeds") != [0, 1]
            or manifest.get("paired_initial_state_and_observations") is not True
            or manifest.get("state_hashes_before") != manifest.get("state_hashes_after")
            or manifest.get("forbidden_calls") != []
            or manifest.get("mutation_guard_restored") is not True
            or manifest.get("fixed_cases_only") is not True
            or manifest.get("random_success_rate_inferred") is not False):
        raise ValueError("Source L0 route manifest is missing or not fixed/zero-update")
    if (set((case["learner"], case["seed"], case["speed_mps"]) for case in cases)
            != {(learner, seed, speed) for learner in ("student", "teacher")
                for seed in (0, 1) for speed in (0.2, 0.6, 1.0)}):
        raise ValueError("L0 route teacher/student/speed/seed matrix is incomplete")
    steps = manifest["observed_steps"]
    dt_s = manifest.get("dt_s")
    if (not isinstance(steps, int) or steps <= 0
            or not isinstance(dt_s, (float, int)) or not 0 < dt_s <= .1):
        raise ValueError("Fixed route duration or control period is invalid")
    expected = {"command": (steps, 12, 3), "root_local": (steps, 12, 3),
                "wheel_force_z": (steps, 12, 4), "joint_request": (steps, 12, 16),
                "joint_target": (steps, 12, 16), "joint_actual": (steps, 12, 16),
                "joint_torque": (steps, 12, 16), "joint_speed": (steps, 12, 16),
                "leg_clearance_m": (steps, 12), "raw_action": (steps, 12, 16),
                "dwell_s": (steps, 12),
                **{name: (steps, 12) for name in ("seen_slope", "seen_stair",
                                                   "full_body_platform", "success",
                                                   "physical", "outside")}}
    for name, shape in expected.items():
        value = arrays[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Incomplete or nonfinite L0 route trace: {name}")
    if not (arrays["wheel_force_z"] > 5).any():
        raise ValueError("Fixed source route has no positive terrain-filtered wheel contact")
    for local, case in enumerate(cases):
        if case["id"] != local or case["level"] != 0:
            raise ValueError("L0 route case IDs or levels changed")
        until = case["end_step"] + 1
        if not 1 <= until <= steps:
            raise ValueError("L0 route case has invalid duration")
        for key, trace in (("seen_slope", "seen_slope"),
                           ("seen_stair", "seen_stair"),
                           ("full_body_platform", "full_body_platform"),
                           ("two_second_success", "success"),
                           ("physical", "physical"), ("outside", "outside")):
            if case[key] != bool(arrays[trace][:until, local].any()):
                raise ValueError(f"L0 route case {local} contradicts {trace} trace")
        issued = arrays["command"][:until, local]
        if (not np.all(issued[:, 1:] == 0)
                or not np.all(np.isclose(issued[:, 0], 0, atol=1e-5)
                              | np.isclose(issued[:, 0], case["speed_mps"], atol=1e-5))):
            raise ValueError(f"L0 route case {local} command left its fixed speed/stop contract")
        dwell = arrays["dwell_s"][:until, local]
        clearance = arrays["leg_clearance_m"][:until, local]
        clearance_step = int(np.argmin(clearance))
        if (not np.isclose(case["maximum_dwell_s"], dwell.max(), atol=1e-5)
                or not np.isclose(case["minimum_leg_clearance_m"], clearance[clearance_step], atol=1e-5)
                or case["minimum_leg_clearance_step"] != clearance_step
                or not np.isclose(case["final_progress_m"],
                                  arrays["root_local"][until - 1, local, 0], atol=1e-5)):
            raise ValueError(f"L0 route case {local} summary differs from its trace")
        if (case["two_second_success"]
                and (case["physical"] or case["outside"] or not case["seen_slope"]
                     or not case["seen_stair"] or not case["full_body_platform"])):
            raise ValueError("L0 route success lacks the whole-route support evidence")
        if case["two_second_success"]:
            success_step = int(np.flatnonzero(arrays["success"][:until, local])[0])
            if (case["end_reason"] != "success"
                    or dwell[success_step] + dt_s * 1e-3 < 2.0
                    or not np.all(arrays["wheel_force_z"][success_step, local] > 5)):
                raise ValueError("L0 route success lacks two-second four-wheel support")
    feasible = [case for case in cases if case["time_feasible"]]
    if len(feasible) != 8 or any(case["speed_mps"] == .2 for case in feasible):
        raise ValueError("0.2 m/s diagnostic was misrepresented as a feasible training case")
    successful = [case for case in feasible if case["two_second_success"]]
    return dict(fixed_cases=12, feasible_cases=8, feasible_successes=len(successful),
                successes_by_learner=dict(Counter(case["learner"] for case in successful)),
                source_l0_eligible_for_behavior_review=bool(successful),
                low_speed_diagnostic_cases=4,
                no_random_success_rate_inferred=True)


def verify_old_nine() -> dict:
    import numpy as np

    final = load_json(OLD / "FINALIZED.json")
    if (final.get("state") != "complete" or final.get("optimizer_updates") != 0
            or final.get("hashes_unchanged") is not True
            or final.get("smoke_only") is not False or final.get("policy_count") != 2):
        raise ValueError("Original fixed-case assessment is incomplete")
    before = load_json(OLD / "HASHES_BEFORE.json")
    if before != {str(SOURCE): SOURCE_SHA256}:
        raise ValueError("Old-nine assessment used a different source")
    protocol = load_json(V21_FIXED_PROTOCOL)["protocol_config"]
    protocol["sources"] = [dict(path=str(SOURCE), completed_updates=SOURCE_COMPLETED)]
    selection = [dict(path=str(SOURCE), label="source_v21", teacher=False),
                 dict(path=str(SOURCE), label="source_v21_teacher", teacher=True)]
    if (load_json(OLD / "protocol.json") != protocol
            or load_json(OLD / "selection.json") != selection):
        raise ValueError("Old-nine assessment did not use the frozen V21 protocol and source pair")
    results = {}
    for suite, count in SUITES.items():
        folder = OLD / suite
        manifest = load_json(folder / "manifest.json")
        cases = manifest.get("cases", [])
        history = manifest.get("history", [])
        if (len(cases) != count
                or manifest.get("smoke_only") is not False
                or manifest.get("optimizer_steps") != 0
                or manifest.get("training_updates") != 0
                or manifest.get("protocol_config") != protocol
                or manifest.get("checkpoint") != str(SOURCE)
                or manifest.get("completed_updates") != SOURCE_COMPLETED
                or manifest.get("stairs15") is not (suite == "stairend_15cm")
                or len(history) != 2
                or any({key: item.get(key) for key in ("path", "label", "teacher")} != expected
                       for item, expected in zip(history, selection))
                or any(case.get("id") != index or case.get("policy") not in (0, 1)
                       or case.get("teacher") is not selection[case["policy"]]["teacher"]
                       or case.get("checkpoint") != str(SOURCE)
                       for index, case in enumerate(cases))
                or sum(case["policy"] == 0 for case in cases) != count // 2
                or not (folder / "traces.npz").is_file()
                or not (folder / "HIP_REVIEW_v2.json").is_file()):
            raise ValueError(f"Incomplete original fixed-case suite: {suite}")
        with np.load(folder / "traces.npz", allow_pickle=False) as traces:
            for name, width in (("body_pos", 3), ("joint_pos", 16), ("command", 3)):
                if name not in traces.files:
                    raise ValueError(f"Missing original fixed-case trace: {suite}/{name}")
                values = traces[name]
                if (values.ndim < 3 or values.shape[0] <= 0
                        or values.shape[1] != count or values.shape[-1] != width
                        or not np.isfinite(values).all()):
                    raise ValueError(f"Incomplete original fixed-case trace: {suite}/{name}")
        results[suite] = dict(cases=count, manifest_sha256=digest(folder / "manifest.json"),
                              traces_sha256=digest(folder / "traces.npz"),
                              hip_review_sha256=digest(folder / "HIP_REVIEW_v2.json"))
    terrain = load_json(OLD / "terrain83/manifest.json")
    present = {case["terrain"] for case in terrain["cases"]}
    if not OLD_TYPES <= present:
        raise ValueError("Original nine terrain families were not all sampled")
    return results


def verify() -> dict:
    import numpy as np
    import torch

    frozen = verify_frozen()
    qualified = load_json(HERE / "QUALIFIED_V25.json")
    if (qualified.get("state") != "verified_not_training"
            or qualified.get("source_sha256") != SOURCE_SHA256
            or qualified.get("optimizer_updates") != 0):
        raise ValueError("Independent Isaac qualification is missing")
    manifest = load_json(ROUTE / "manifest.json")
    with np.load(ROUTE / "traces.npz") as arrays:
        route = verify_route(manifest, arrays)
    old = verify_old_nine()
    before = manifest["state_hashes_before"]
    if not all(re.fullmatch(r"[0-9a-f]{64}", item) for item in before.values()):
        raise ValueError("Fixed route model/optimizer state hashes are malformed")
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    from contract import STATE_KEYS
    from state_guard import state_sha256

    independent = {name: state_sha256(saved[key], torch) for name, key in STATE_KEYS.items()}
    if (before != frozen["source_state_sha256"] or before != independent
            or digest(SOURCE) != SOURCE_SHA256):
        raise ValueError("Fixed route did not retain exact V21 model and optimizer state")
    result = dict(state="fixed_source_diagnostics_verified", source_sha256=SOURCE_SHA256,
                  optimizer_updates=0, route_manifest_sha256=digest(ROUTE / "manifest.json"),
                  route_traces_sha256=digest(ROUTE / "traces.npz"),
                  old_nine_finalized_sha256=digest(OLD / "FINALIZED.json"),
                  qualification_sha256=digest(HERE / "QUALIFIED_V25.json"),
                  v23_assessment_script_sha256=digest(V23_ASSESSMENT / "run_assessment.py"),
                  route=route, original_fixed_suites=old,
                  training_blocked_until_written_behavior_eligibility=True,
                  real_robot_release=False)
    write_new(HERE / "DIAGNOSTICS_VERIFIED.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify()))
