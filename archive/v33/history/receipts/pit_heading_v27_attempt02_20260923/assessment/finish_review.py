"""After the finite V27 run, evaluate once and stop at the behavior gate."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


STAGE = Path(__file__).resolve().parent.parent
HERE = STAGE / "assessment"
ORCH = STAGE / "orchestration"
SOURCE = HERE / "source25_750/FINALIZED.json"
TARGET = ORCH / "training_34839_to_35123/model_35122.pt"
STATUS = HERE / "review_status.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(state, **extra):
    value = dict(state=state, **extra)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, STATUS)


def gpu_busy():
    return bool(subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip())


def main():
    if STATUS.exists() or (HERE / "candidate25_750").exists():
        raise FileExistsError("V27 behavior review has already started")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    status("waiting_for_finite_training")
    deadline = time.monotonic() + 4 * 3600
    try:
        while time.monotonic() < deadline:
            path = ORCH / "pipeline_status.json"
            current = json.loads(path.read_text()) if path.exists() else {}
            if current.get("stage") == "technical_failure":
                raise RuntimeError("Training failed; review will not evaluate an unverified checkpoint")
            if current.get("stage") == "complete_pending_behavior_review":
                break
            time.sleep(20)
        else:
            raise TimeoutError("Finite training did not complete in four hours")
        if (current.get("completed_updates") != 35123
                or not (ORCH / "training_34839_to_35123/SUCCESS.json").is_file()
                or digest(TARGET) != current.get("candidate_sha256")):
            raise RuntimeError("Finite training checkpoint or status is inconsistent")
        sys.path.insert(0, str(ORCH))
        from preflight import verify_frozen

        verify_frozen()
        while gpu_busy() and time.monotonic() < deadline:
            time.sleep(10)
        if gpu_busy():
            raise TimeoutError("GPU remained occupied after training")
        output = HERE / "candidate25_750"
        status("evaluating_25cm", candidate_sha256=digest(TARGET))
        command = [sys.executable, "-u", str(HERE / "evaluate_pit25.py"),
                   "--source", str(TARGET), "--source-sha256", digest(TARGET),
                   "--completed", "35123", "--output", str(output),
                   "--steps", "750"]
        with (HERE / "candidate25_750.log").open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not (output / "FINALIZED.json").is_file():
            raise RuntimeError("Candidate evaluation finished without zero-update evidence")
        review = HERE / "PIT25_BEHAVIOR_GATE.json"
        command = [sys.executable, str(HERE / "review_pit25.py"),
                   "--source", str(SOURCE),
                   "--candidate", str(output / "FINALIZED.json"),
                   "--output", str(review)]
        with (HERE / "review_pit25.log").open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        passed = json.loads(review.read_text())["pass"]
        status("behavior_gate_passed" if passed else "behavior_gate_rejected",
               completed_updates=35123, candidate_sha256=digest(TARGET),
               automatic_extension=False)
    except Exception as exc:
        status("technical_failure", error=str(exc), automatic_retry=False)
        raise


if __name__ == "__main__":
    main()
