"""Single locked zero-update contact diagnosis; original frozen sources retained."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "stair_release_validation_attempt02_20260922"
V21 = ROOT / "stair_coverage_v21"
BASE = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")


def hashes():
    paths = [V21 / "training_33881_to_34823/model_34822.pt",
             SOURCE / "export_candidate/thunder_cts_moe_34823.onnx",
             SOURCE / "evaluate_stairs_onnx_v3.py", *sorted((V21 / "code").glob("*.py"))]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    with (BASE / "full_episode_v12_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU occupied")
        before = hashes()
        assert before[str(V21 / "training_33881_to_34823/model_34822.pt")] == "521a31d1acfc7e0906f4ac3260ee30f161af5b1c18278367d0c0045b7afe3fa5"
        (HERE / "HASHES_BEFORE.json").write_text(json.dumps(before, indent=2))
        for condition in ("nominal", "perturbed"):
            (HERE / "STATUS.json").write_text(json.dumps(dict(state="running", condition=condition, optimizer_updates=0)))
            command = [sys.executable, "-u", str(HERE / "contact_cause_probe.py"),
                       "--root", str(BASE), "--code", str(V21 / "code"), "--output", str(HERE / condition),
                       "--selection", str(SOURCE / "results/selection.json"),
                       "--protocol", str(SOURCE / "results/protocol.json"), "--suite", "stairend",
                       "--onnx", str(SOURCE / "export_candidate/thunder_cts_moe_34823.onnx")]
            if condition == "perturbed":
                command.append("--perturbed")
            with (HERE / (condition + ".log")).open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert before == hashes(), "Frozen source changed"
        (HERE / "FINALIZED.json").write_text(json.dumps(dict(state="complete", optimizer_updates=0,
                                                           hashes_unchanged=True, trajectories_exact=True)))
        with zipfile.ZipFile(HERE / "CONTACT_REVIEW.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            archive.write(HERE / "FINALIZED.json", "FINALIZED.json")
            for condition in ("nominal", "perturbed"):
                for name in ("CONTACT_CAPTURE.json", "CONTACT_EVENTS.json", "CONTACT_CAUSE_5MS.npz", "stair_profiles.npz"):
                    archive.write(HERE / condition / name, condition + "/" + name)
        (HERE / "STATUS.json").write_text(json.dumps(dict(state="complete", optimizer_updates=0)))


if __name__ == "__main__":
    main()
