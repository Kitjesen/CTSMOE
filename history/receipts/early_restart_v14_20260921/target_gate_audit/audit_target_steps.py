"""Reproduce the V14 target-step gate audit from frozen evaluation artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from target_step_mask import observable_target_deltas


TARGET_STEP_LIMIT = 0.2401
SUITES = ("terrain83", "flat25")


def _peak(
    target: np.ndarray,
    initial_q: np.ndarray,
    *,
    end_step: int,
    terms: list[str],
) -> dict[str, Any] | None:
    deltas, step_ids = observable_target_deltas(
        target, initial_q, end_step=end_step, termination_terms=terms
    )
    if deltas.size == 0:
        return None
    flat_index = int(np.abs(deltas).argmax())
    row, joint = np.unravel_index(flat_index, deltas.shape)
    before = initial_q[joint] if row == 0 else target[step_ids[row - 1], joint]
    after = target[step_ids[row], joint]
    return {
        "peak_rad": float(abs(deltas[row, joint])),
        "step": int(step_ids[row]),
        "joint_column": int(joint),
        "before_rad": float(before),
        "after_rad": float(after),
    }


def audit_suite(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    analysis = json.loads((directory / "analysis_v14.json").read_text(encoding="utf-8"))
    leg_ids = np.asarray(manifest["request_joint_ids"], dtype=np.int64)
    joint_names = np.asarray(manifest["joint_names"])

    with np.load(directory / "traces.npz") as traces:
        target = traces["target"][:, :, leg_ids]
        joint_pos = traces["joint_pos"][:, :, leg_ids]
        initial_q = traces["initial_q"][:, leg_ids]

    rows: list[dict[str, Any]] = []
    terminal_target_joint_pos_errors: list[float] = []
    for case in analysis["cases"]:
        case_id = int(case["id"])
        end_step = int(case["end_step"])
        terms = list(case.get("terms") or [])
        case_target = target[:, case_id]

        raw_peak = _peak(
            case_target, initial_q[case_id], end_step=end_step, terms=[]
        )
        observable_peak = _peak(
            case_target, initial_q[case_id], end_step=end_step, terms=terms
        )
        if terms:
            terminal_target_joint_pos_errors.append(
                float(np.abs(target[end_step, case_id] - joint_pos[end_step, case_id]).max())
            )

        raw_over = raw_peak is not None and raw_peak["peak_rad"] > TARGET_STEP_LIMIT
        observable_over = (
            observable_peak is not None
            and observable_peak["peak_rad"] > TARGET_STEP_LIMIT
        )
        row = {
            "id": case_id,
            "label": case["label"],
            "terrain": case["terrain"],
            "level": case["level"],
            "mode": case["mode"],
            "direction": case["direction"],
            "velocity": case["velocity"],
            "end_step": end_step,
            "terms": terms,
            "raw_peak": raw_peak,
            "observable_peak": observable_peak,
            "raw_over_limit": raw_over,
            "observable_over_limit": observable_over,
        }
        if raw_peak is not None:
            row["raw_peak"]["joint"] = str(
                joint_names[leg_ids[raw_peak["joint_column"]]]
            )
            row["raw_peak_at_end_step"] = raw_peak["step"] == end_step
        if observable_peak is not None:
            row["observable_peak"]["joint"] = str(
                joint_names[leg_ids[observable_peak["joint_column"]]]
            )
        rows.append(row)

    raw_over_rows = [row for row in rows if row["raw_over_limit"]]
    observable_over_rows = [row for row in rows if row["observable_over_limit"]]
    labels = sorted({row["label"] for row in rows})
    max_by_label = {
        label: max(
            row["observable_peak"]["peak_rad"]
            for row in rows
            if row["label"] == label and row["observable_peak"] is not None
        )
        for label in labels
    }

    samples = sorted(
        (row for row in raw_over_rows if row["label"].startswith("V14_")),
        key=lambda row: row["raw_peak"]["peak_rad"],
        reverse=True,
    )[:8]
    sample_fields = (
        "id",
        "label",
        "terrain",
        "level",
        "mode",
        "direction",
        "velocity",
        "end_step",
        "terms",
        "raw_peak",
    )

    return {
        "suite": directory.name,
        "case_count": len(rows),
        "dt_s": manifest["dt"],
        "leg_target_rate_rad_s": manifest["actuator_limits"]["leg_target_rate_rad_s"],
        "nominal_target_step_rad": (
            manifest["dt"] * manifest["actuator_limits"]["leg_target_rate_rad_s"]
        ),
        "target_step_limit_rad": TARGET_STEP_LIMIT,
        "raw_over_limit_count": len(raw_over_rows),
        "raw_over_limit_source_count": sum(
            not row["label"].startswith("V14_") for row in raw_over_rows
        ),
        "raw_over_limit_candidate_count": sum(
            row["label"].startswith("V14_") for row in raw_over_rows
        ),
        "raw_over_limit_by_label": dict(Counter(row["label"] for row in raw_over_rows)),
        "raw_over_limit_at_end_step_count": sum(
            row.get("raw_peak_at_end_step", False) for row in raw_over_rows
        ),
        "raw_over_limit_with_termination_terms_count": sum(
            bool(row["terms"]) for row in raw_over_rows
        ),
        "observable_over_limit_count": len(observable_over_rows),
        "observable_max_by_label_rad": max_by_label,
        "termination_case_count": len(terminal_target_joint_pos_errors),
        "termination_target_equals_joint_pos_count": sum(
            error <= 1e-7 for error in terminal_target_joint_pos_errors
        ),
        "termination_target_joint_pos_max_error_rad": max(
            terminal_target_joint_pos_errors, default=None
        ),
        "lost_pre_reset_terminal_target_count": len(terminal_target_joint_pos_errors),
        "candidate_raw_over_limit_samples": [
            {field: row[field] for field in sample_fields} for row in samples
        ],
    }


def audit_gate(path: Path) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    reason_counts = Counter(reason["code"] for reason in gate["reasons"])
    target_count = reason_counts.get("target_step_limit", 0)
    return {
        "accepted": gate["accepted"],
        "action": gate["action"],
        "reason_count": len(gate["reasons"]),
        "reason_counts": dict(reason_counts),
        "target_step_reason_count": target_count,
        "independent_non_target_reason_count": len(gate["reasons"]) - target_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_receipt = Path(__file__).resolve().parent.parent
    parser.add_argument("--receipt", type=Path, default=default_receipt)
    args = parser.parse_args()

    result = {
        "scope": "offline diagnostic; terminal pre-reset action targets are unavailable",
        "suites": [
            audit_suite(args.receipt / "evaluation300" / suite) for suite in SUITES
        ],
        "gate": audit_gate(args.receipt / "gate300.json"),
        "interpretation": {
            "observed_nonterminal_target_limit_violation": False,
            "terminal_pre_reset_target_recovered": False,
            "offline_mask_authorizes_gate_acceptance": False,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
