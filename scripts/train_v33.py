"""Explicit portable command builder for the unchanged V33 source snapshot."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, default=repo / "vendor")
    p.add_argument("--updates", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.updates <= 0:
        p.error("updates must be positive")
    if a.output.exists():
        p.error("output must be a new directory")
    spec = importlib.util.spec_from_file_location(
        "frozen_launch",
        repo
        / "history/receipts/continuous_training_v33_20260924/orchestration/launch_common.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE = a.runtime_root.resolve()
    command = [
        sys.executable,
        str(repo / "src/thunder_v33/train.py"),
        *module.args(a.checkpoint.resolve(), a.output.resolve(), a.updates),
    ]
    if a.dry_run:
        import json

        print(json.dumps(command, indent=2))
        return
    if not a.checkpoint.is_file():
        p.error("checkpoint is missing")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
