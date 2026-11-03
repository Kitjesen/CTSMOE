"""Offline, zero-update audit of mirrored hip requests, targets, and positions.

Supports both the V21 selected-case archive and the batched evaluation archives.
The source trajectories and training checkpoints are read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


INWARD_SIGNS = {
    "fl_hip_joint": 1.0,
    "fr_hip_joint": -1.0,
    "rl_hip_joint": -1.0,
    "rr_hip_joint": 1.0,
}
HIP_SOFT_LIMIT_RAD = 0.369
HIP_HARD_LIMIT_RAD = 0.41
SATURATION_ATOL_RAD = 0.002
NEAR_HARD_MARGIN_RAD = 0.01
CONTACT_WINDOW_S = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(series: np.ndarray) -> dict[str, float]:
    return {
        "p50_rad": float(np.quantile(series, 0.50)),
        "p95_rad": float(np.quantile(series, 0.95)),
        "peak_rad": float(np.max(series)),
    }


def _duration(mask: np.ndarray, dt: float) -> dict[str, float]:
    if mask.ndim != 1:
        raise ValueError("Expected one time axis")
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return {
        "total_s": float(np.count_nonzero(mask) * dt),
        "longest_contiguous_s": float(np.max(ends - starts + 1) * dt) if len(starts) else 0.0,
    }


def _validate_series(series: dict[str, np.ndarray], names: list[str], request_ids: list[int], dt: float) -> int:
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if len(names) != len(set(names)) or not set(INWARD_SIGNS).issubset(names):
        raise ValueError("Four unique hip names are required")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_joint_ids contains duplicates")
    length = len(series["joint_pos"])
    if length == 0:
        raise ValueError("Empty active trajectory")
    for key in ("joint_pos", "target", "requested"):
        value = series[key]
        width = len(request_ids) if key == "requested" else len(names)
        if value.shape != (length, width) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} trajectory shape or values: {value.shape}")
    if any(names.index(name) not in request_ids for name in INWARD_SIGNS):
        raise ValueError("Missing hip request mapping")
    for key, value in series.items():
        if len(value) != length or not np.isfinite(value).all():
            raise ValueError(f"Invalid optional {key} trajectory")
    return length


def _optional_channels(series: dict[str, np.ndarray], time_index: int, dt: float) -> dict[str, object]:
    result: dict[str, object] = {}
    left = max(0, time_index - int(round(CONTACT_WINDOW_S / dt)))
    right = min(len(series["joint_pos"]), time_index + int(round(CONTACT_WINDOW_S / dt)) + 1)
    for channel, field in (
        ("calf_contact_peak", "calf_terrain_contact_peak_N"),
        ("base_contact", "base_contact_peak_N"),
        ("wheel_ground_force_z", "wheel_ground_support_peak_N"),
    ):
        if channel in series:
            result[field] = float(np.max(np.abs(series[channel][left:right])))
    for channel in ("torque_peak", "speed_peak"):
        result[f"{channel}_available"] = channel in series
    result["self_collision_force_available"] = False
    result["contact_window_s"] = CONTACT_WINDOW_S
    return result


def analyze_case(
    series: dict[str, np.ndarray], meta: dict[str, object], *, soft_limit_rad: float = HIP_SOFT_LIMIT_RAD
) -> dict[str, object]:
    names = list(meta["joint_names"])
    request_ids = list(meta["request_joint_ids"])
    defaults = np.asarray(meta["default_q"], dtype=float)
    dt = float(meta["dt"])
    if defaults.shape != (len(names),) or not np.isfinite(defaults).all():
        raise ValueError("Invalid default joint angles")
    if not 0 < soft_limit_rad < 1:
        raise ValueError("Invalid hip soft limit")
    count = _validate_series(series, names, request_ids, dt)
    case: dict[str, object] = {
        "case_id": meta["id"],
        "condition": meta.get("condition"),
        "policy": meta.get("policy"),
        "terrain": meta.get("terrain"),
        "level": meta.get("level"),
        "timebase": {"control_dt_s": dt, "control_samples": count, "duration_s": count * dt},
        "channels_available": sorted(series),
        "unavailable": [
            "self-collision force and minimum leg-to-leg clearance",
            "200 Hz joint angle and command trajectory (these traces are control-step samples)",
        ],
        "hips": {},
    }
    request_excess_per_step = np.zeros(count, dtype=float)
    for name, sign in INWARD_SIGNS.items():
        joint_index = names.index(name)
        request_index = request_ids.index(joint_index)
        request = series["requested"][:, request_index] * sign
        request_excess_per_step += np.maximum(request - soft_limit_rad, 0.0)
        target = series["target"][:, joint_index] * sign
        actual = series["joint_pos"][:, joint_index] * sign
        default_inward = float(defaults[joint_index] * sign)
        requested_over_limit = request > soft_limit_rad + SATURATION_ATOL_RAD
        target_at_limit = target >= soft_limit_rad - SATURATION_ATOL_RAD
        # Target records actuator delivery after delay/slew. Same-frame overlap
        # is evidence of saturation, but cannot prove the exact request-to-target
        # causal edge without the delayed action queue.
        saturation_overlap = requested_over_limit & target_at_limit
        actual_over_limit = actual > soft_limit_rad + SATURATION_ATOL_RAD
        peak_index = int(np.argmax(actual))
        hip: dict[str, object] = {
            "inward_sign": sign,
            "default_inward_rad": default_inward,
            "soft_inward_limit_rad": soft_limit_rad,
            "requested_inward": _quantiles(request),
            "limited_target_inward": _quantiles(target),
            "actual_inward": _quantiles(actual),
            "limited_target_minus_actual_inward": {
                "mean_rad": float(np.mean(target - actual)),
                "p50_rad": float(np.quantile(target - actual, 0.50)),
                "p95_rad": float(np.quantile(target - actual, 0.95)),
                "minimum_rad": float(np.min(target - actual)),
                "maximum_rad": float(np.max(target - actual)),
            },
            "durations": {
                "requested_beyond_soft_limit": _duration(requested_over_limit, dt),
                "target_at_soft_limit": _duration(target_at_limit, dt),
                "request_and_limited_target_saturation_overlap": _duration(saturation_overlap, dt),
                "actual_beyond_soft_limit": _duration(actual_over_limit, dt),
                "actual_near_hard_limit_within_001_rad": _duration(
                    actual >= HIP_HARD_LIMIT_RAD - NEAR_HARD_MARGIN_RAD, dt
                ),
                "actual_beyond_default_pose_not_failure": _duration(actual > default_inward + 0.001, dt),
            },
            "peak_target_tracking_error_rad": float(np.max(np.abs(actual - target))),
            "at_actual_inward_peak": {
                "time_s": peak_index * dt,
                "requested_inward_rad": float(request[peak_index]),
                "target_inward_rad": float(target[peak_index]),
                "actual_inward_rad": float(actual[peak_index]),
                **_optional_channels(series, peak_index, dt),
            },
        }
        for channel in ("torque_peak", "speed_peak"):
            if channel in series:
                hip[f"{channel}_at_control_samples"] = {
                    "mean": float(np.mean(series[channel][:, joint_index])),
                    "p95": float(np.quantile(series[channel][:, joint_index], 0.95)),
                    "peak": float(np.max(series[channel][:, joint_index])),
                    "at_actual_inward_peak": float(series[channel][peak_index, joint_index]),
                }
        if "joint_vel" in series:
            hip["signed_inward_joint_velocity_rad_s"] = {
                "mean": float(np.mean(series["joint_vel"][:, joint_index] * sign)),
                "p95": float(np.quantile(series["joint_vel"][:, joint_index] * sign, 0.95)),
            }
        stages = {}
        stage_source = "stair_platform_hull" if "stair_platform_hull" in series else (
            "platform_stop" if "platform_stop" in series else None
        )
        if stage_source is not None:
            platform = series[stage_source].astype(bool)
            for label, mask in (("before_platform", ~platform), ("on_or_after_platform", platform)):
                if mask.any():
                    stages[label] = {
                        "samples": int(mask.sum()),
                        "actual_inward_p95_rad": float(np.quantile(actual[mask], 0.95)),
                        "saturation_overlap_duration_s": float(np.count_nonzero(saturation_overlap & mask) * dt),
                    }
        hip["stage_source"] = stage_source
        hip["stage_summary"] = stages
        case["hips"][name] = hip
    case["four_hip_request_excess_above_soft_limit"] = {
        "definition": "sum(max(inward_sign*raw_request - 0.369 rad, 0)) over four hips, at each control step",
        "mean_rad_per_step": float(np.mean(request_excess_per_step)),
        "p95_rad_per_step": float(np.quantile(request_excess_per_step, 0.95)),
        "peak_rad_per_step": float(np.max(request_excess_per_step)),
        "any_excess_duration_s": float(np.count_nonzero(request_excess_per_step > 0) * dt),
    }
    return case


def _prefixed_cases(archive: np.lib.npyio.NpzFile, metadata: list[dict[str, object]]) -> list[dict[str, object]]:
    cases = []
    for meta in metadata:
        prefix = f"{meta['condition']}_{meta['id']}__"
        series = {key.removeprefix(prefix): archive[key] for key in archive.files if key.startswith(prefix)}
        cases.append(analyze_case(series, meta))
    return cases


def _batched_cases(archive: np.lib.npyio.NpzFile, metadata: dict[str, object]) -> list[dict[str, object]]:
    required = ("joint_pos", "target", "requested")
    data = {key: archive[key] for key in archive.files if key in required or key in (
        "active", "platform_stop", "stair_platform_hull", "base_contact", "calf_contact_peak",
        "wheel_ground_force_z", "torque_peak", "speed_peak", "joint_vel"
    )}
    if not all(key in data for key in required):
        raise ValueError("Batched archive missing hip trajectories")
    total_steps, count = data["joint_pos"].shape[:2]
    ids = archive["case_ids"].tolist() if "case_ids" in archive else list(range(count))
    if len(ids) != count:
        raise ValueError("case_ids does not match trajectory count")
    lookup = {int(case["id"]): case for case in metadata["cases"]}
    results = []
    for index, case_id in enumerate(ids):
        if case_id not in lookup:
            raise ValueError(f"No manifest entry for case {case_id}")
        meta = dict(metadata, **lookup[case_id])
        meta["id"] = case_id
        if "default_q" in archive:
            meta["default_q"] = archive["default_q"].tolist()
        if "request_joint_ids" in archive:
            meta["request_joint_ids"] = archive["request_joint_ids"].tolist()
        if "dt" in archive:
            meta["dt"] = float(archive["dt"])
        active = data.get("active", np.ones((total_steps, count), dtype=bool))[:, index].astype(bool)
        if active.ndim != 1 or not active.any() or not np.array_equal(active, np.arange(total_steps) < active.sum()):
            raise ValueError(f"Case {case_id}: active samples must form one prefix")
        length = int(active.sum())
        series = {key: values[:length, index] for key, values in data.items() if key != "active"}
        results.append(analyze_case(series, meta))
    return results


def analyze_archive(trace: Path, manifest: Path) -> dict[str, object]:
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    with np.load(trace, allow_pickle=False) as archive:
        if isinstance(metadata, list):
            cases = _prefixed_cases(archive, metadata)
            fmt = "prefixed_selected_cases"
        elif isinstance(metadata, dict):
            cases = _batched_cases(archive, metadata)
            fmt = "batched_cases"
        else:
            raise ValueError("Manifest must be a case list or a case-containing object")
    return {
        "schema": "hip_inward_time_audit_v1",
        "input": {"trace": str(trace), "trace_sha256": _sha256(trace),
                  "manifest": str(manifest), "manifest_sha256": _sha256(manifest), "format": fmt},
        "interpretation": [
            "Positive mirrored angle means inward motion on every hip.",
            "Samples and duration are at the policy control rate; 200 Hz peak channels, when present, are aggregate peaks inside each control step.",
            "Above-default duration measures posture departure, not a physical-limit violation.",
            "Contact coincidence alone cannot establish collision as the cause of inward motion.",
            "Request/target same-frame overlap is not an exact clipping event because targets can be delayed and slew limited.",
            "These selected/fixed trajectories are not a random success rate or a hardware fault diagnosis.",
        ],
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_archive(args.trace, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
