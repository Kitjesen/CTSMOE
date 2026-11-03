"""Pause receipt, zero-update recording, and exact remaining-budget continuation."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import traceback
STAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE / "orchestration"))
from launch_common import LOCK, frozen, digest
from driver import run, train, status, TARGET
import torch

def main():
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU occupied")
        frozen()
        source = max((STAGE/"continuation").glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        start = torch.load(source, map_location="cpu", weights_only=False)["completed_updates"]
        assert 45000 <= start < TARGET
        receipt = dict(source=str(source), sha256=digest(source), completed=start, target=TARGET, remaining=TARGET-start)
        (STAGE/"recording_request/SOURCE_v2.json").write_text(json.dumps(receipt, indent=2))
        run(STAGE/"recording_request/record_long_v2.py", ["--checkpoint", str(source), "--output", str(STAGE/"recording_request/flat_long")], "flat_long_recording_v2")
        proof=json.loads((STAGE/"recording_request/flat_long/FINALIZED.json").read_text())
        assert proof["zero_optimizer_updates"] and proof["state_hashes_unchanged"] and digest(source)==receipt["sha256"]
        checkpoint=train(source, STAGE/"continuation_after_flat_recording", TARGET-start, "training_after_flat_recording")
        run(STAGE/"assessment/assess_v33.py", ["--checkpoint", str(checkpoint), "--output", str(STAGE/"final_assessment")], "final_assessment")
        status("complete_pending_review", completed=TARGET, checkpoint=str(checkpoint), hardware_release=False)

if __name__=="__main__":
    try:
        main()
    except Exception:
        status("technical_failure", error=traceback.format_exc())
        raise
