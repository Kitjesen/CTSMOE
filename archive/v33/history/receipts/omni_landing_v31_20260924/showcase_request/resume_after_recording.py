"""Resume the unchanged V31 budget after the user's sequential video request."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys

STAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE / "orchestration"))
from v31_common import frozen, args, LOCK, digest  # noqa: E402


def main():
    import torch

    here = Path(__file__).parent
    status = here / "RESUME_STATUS.json"
    if status.exists():
        raise FileExistsError(status)

    def write(state, **kwargs):
        status.write_text(
            json.dumps(dict(state=state, target=46194, **kwargs), indent=2)
        )

    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen()
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip():
            raise RuntimeError("GPU is occupied")
        result = json.loads((here / "recording_v2/FINALIZED.json").read_text())
        assert result["zero_optimizer_updates"] and result["state_hashes_unchanged"]
        checkpoint = Path(result["checkpoint"])
        assert digest(checkpoint) == result["sha256"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        start = saved["completed_updates"]
        assert (
            start == 38744
            and "optimizer_state_dict" in saved
            and "optimizer_stu_enc_state_dict" in saved
        )
        remaining = 46194 - start
        output = STAGE / f"training_{start}_to_46194"
        if output.exists():
            raise FileExistsError(output)
        try:
            with (here / "resumed_training.log").open("x") as log:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(STAGE / "code/train.py"),
                        *args(checkpoint, output, remaining),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                write(
                    "training",
                    pid=proc.pid,
                    start=start,
                    remaining=remaining,
                    reason="resume after zero-update requested videos",
                )
                if proc.wait():
                    raise RuntimeError(
                        "Training exit; preserve checkpoint and investigate"
                    )
            success = json.loads((output / "SUCCESS.json").read_text())
            assert success["completed_updates"] == 46194
            write("evaluating_final", completed=46194)
            evaluation = STAGE / "assessment/candidate"
            with (here / "final_assessment.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(STAGE / "assessment/assess_v31.py"),
                        "--checkpoint",
                        str(output / "model_46193.pt"),
                        "--output",
                        str(evaluation),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            candidate = json.loads((evaluation / "FINALIZED.json").read_text())
            assert (
                candidate["state_hashes_unchanged"]
                and candidate["optimizer_updates"] == 0
            )
            (STAGE / "assessment/COMPARISON.json").write_text(
                json.dumps(
                    dict(
                        source=json.loads(
                            (STAGE / "assessment/source/FINALIZED.json").read_text()
                        ),
                        candidate=candidate,
                    ),
                    indent=2,
                )
            )
            write("complete_pending_review", completed=46194, hardware_release=False)
        except Exception as exc:
            write("technical_failure", error=str(exc), start=start)
            raise


if __name__ == "__main__":
    main()
