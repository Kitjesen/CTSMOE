"""V20 bounded continuation; configuration-driven, no implicit budget expansion."""

from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASELINE = 31831
TARGETS = (34823, 37823, 41823)
LOST = 8


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def budget(completed):
    spent = completed - BASELINE + LOST
    if not 1008 <= spent <= 10000:
        raise ValueError("budget exceeded")
    return spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    c = json.loads(ap.parse_args().config.read_text(encoding="utf-8"))
    root, out, code = map(Path, (c["root"], c["output"], c["code"]))
    source, completed = Path(c["source_path"]), int(c["source_completed"])
    if budget(completed) > 10000 or completed >= TARGETS[-1]:
        raise SystemExit("invalid source budget")
    if sha256(source) != c["source_sha256"]:
        raise SystemExit("source SHA mismatch")
    if (out / "STARTED_V20.json").exists():
        raise SystemExit("duplicate V20 start")
    with (root / "pipeline.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True, exist_ok=False)
        (out / "STARTED_V20.json").write_text(
            json.dumps(
                {"source_completed": completed, "source_sha256": c["source_sha256"]}
            ),
            encoding="utf-8",
        )
        for argv_key in ("preflight_argv", "verify_argv"):
            argv = [
                str(x)
                .replace("{source}", str(source))
                .replace("{code}", str(code))
                .replace("{output}", str(out))
                for x in c[argv_key]
            ]
            if subprocess.run(argv, check=False).returncode:
                raise RuntimeError(argv_key + " failed")
        ready = json.loads(Path(c["ready_path"]).read_text(encoding="utf-8"))
        if (
            ready.get("state") != c.get("ready_state", "verified_not_training")
            or ready.get("source_completed") != completed
        ):
            raise RuntimeError("invalid independent READY")
        for target in TARGETS:
            if target <= completed:
                continue
            inc = target - completed
            if budget(target) > 10000:
                raise RuntimeError("target exceeds budget")
            stage = out / f"training_{completed}_to_{target}"
            with (Path(c["base"]) / "full_episode_v12_pipeline.lock").open("a") as gpu:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if stage.exists() or shutil.disk_usage(out).free < 10 * 1024**3:
                    raise RuntimeError("unsafe stage")
                args = [
                    str(x)
                    .replace("{source}", str(source))
                    .replace("{code}", str(code))
                    .replace("{output}", str(stage))
                    .replace("{updates}", str(inc))
                    for x in c["train_argv"]
                ]
                if subprocess.run(args, check=False).returncode:
                    raise RuntimeError("training failed")
            source, completed = stage / f"model_{target - 1}.pt", target
            eval_args = [
                str(x)
                .replace("{source}", str(source))
                .replace("{completed}", str(completed))
                .replace("{output}", str(out / f"evaluation_{target}"))
                for x in c["evaluate_argv"]
            ]
            if subprocess.run(eval_args, check=False).returncode:
                raise RuntimeError("evaluation failed")
            if budget(completed) >= 3000 and c.get("gate_argv"):
                if subprocess.run(c["gate_argv"], check=False).returncode:
                    raise RuntimeError("gate stopped continuation")
    (out / "COMPLETE.json").write_text(
        json.dumps({"completed": completed, "spent": budget(completed)}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
