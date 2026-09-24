"""Manual, review-gated V25 entry; never called by qualification/diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from contract import (BASE, HERE, MAX_V25_UPDATES, SOURCE, SOURCE_SHA256,
                      STAGE, budget, digest, load_json, training_args,
                      verify_frozen, write_new)


REVIEW = HERE / "BEHAVIOR_ELIGIBILITY.json"
DIAGNOSTICS = HERE / "DIAGNOSTICS_VERIFIED.json"
QUALIFIED = HERE / "QUALIFIED_V25.json"
PLAN = HERE / "TRAINING_ENTRY.json"


def validate_eligibility(review: dict, diagnostics: dict) -> None:
    if (review.get("decision") != "eligible_for_bounded_v25_training"
            or review.get("source_path") != str(SOURCE)
            or review.get("source_sha256") != SOURCE_SHA256
            or review.get("qualification_sha256") != digest(QUALIFIED)
            or review.get("diagnostics_sha256") != digest(DIAGNOSTICS)
            or review.get("real_robot_release") is not False
            or review.get("automatic_training") is not False
            or diagnostics.get("route", {}).get("feasible_successes", 0) <= 0
            or not isinstance(review.get("route_findings"), str)
            or len(review["route_findings"].strip()) < 80
            or not isinstance(review.get("old_nine_findings"), str)
            or len(review["old_nine_findings"].strip()) < 80
            or review.get("leg_clearance_reviewed") is not True
            or not isinstance(review.get("prior_budget_evidence"), str)
            or len(review["prior_budget_evidence"].strip()) < 40
            or not isinstance(review.get("prior_global_spent_updates"), int)
            or not 4000 <= review["prior_global_spent_updates"] <= 5500):
        raise ValueError("Written route/old-nine behavior and global-budget eligibility is absent")


def prepare_entry(updates: int, output: Path) -> dict:
    verify_frozen()
    if not 1 <= updates <= MAX_V25_UPDATES:
        raise ValueError("V25 finite training must request 1..500 updates")
    if not output.is_absolute() or output.exists():
        raise ValueError("Training output must be a new absolute directory")
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError("V21 source changed")
    diagnostics = load_json(DIAGNOSTICS)
    if (diagnostics.get("state") != "fixed_source_diagnostics_verified"
            or diagnostics.get("source_sha256") != SOURCE_SHA256
            or diagnostics.get("optimizer_updates") != 0
            or diagnostics.get("route", {}).get("source_l0_eligible_for_behavior_review") is not True):
        raise ValueError("Source L0 failed all feasible fixed cases or diagnostics are incomplete")
    review = load_json(REVIEW)
    validate_eligibility(review, diagnostics)
    accounting = budget(updates)
    accounting["prior_global_spent_confirmed"] = review["prior_global_spent_updates"]
    accounting["remaining_from_confirmed_spend"] = (
        accounting["original_total_updates"] - review["prior_global_spent_updates"] - updates)
    args = training_args(SOURCE, output, updates)
    plan = dict(state="prepared_for_manual_execution", source=str(SOURCE),
                source_sha256=SOURCE_SHA256, output=str(output), updates=updates,
                command=[sys.executable, "-u", str(STAGE / "code/train.py"), *args],
                qualification_sha256=digest(QUALIFIED),
                diagnostics_sha256=digest(DIAGNOSTICS),
                behavior_eligibility_sha256=digest(REVIEW), budget=accounting,
                automatic_training=False, automatic_extension=False,
                real_robot_release=False)
    if PLAN.exists():
        if load_json(PLAN) != plan:
            raise ValueError("A different V25 training entry is already prepared")
    else:
        write_new(PLAN, plan)
    return plan


def execute_prepared(plan: dict) -> None:
    import fcntl

    if (plan.get("state") != "prepared_for_manual_execution"
            or (HERE / "STARTED_TRAINING.json").exists()):
        raise FileExistsError("Training was already attempted or entry is incomplete")
    with (BASE / "full_episode_v12_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip():
            raise RuntimeError("GPU already occupied")
        write_new(HERE / "STARTED_TRAINING.json", dict(
            plan_sha256=digest(PLAN), source_sha256=SOURCE_SHA256,
            requested_updates=plan["updates"], output=plan["output"],
            automatic_extension=False, real_robot_release=False))
        subprocess.run(plan["command"], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Explicit manual launch of the previously prepared plan")
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("Training entry assertions must remain enabled")
    plan = prepare_entry(args.updates, args.output)
    if args.execute:
        execute_prepared(plan)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
