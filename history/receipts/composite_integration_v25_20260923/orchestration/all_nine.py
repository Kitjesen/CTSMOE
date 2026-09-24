"""One-shot source-only fixed-case baseline for all original terrain types."""

from __future__ import annotations

import subprocess
import sys

from contract import (HERE, SOURCE, SOURCE_COMPLETED, SOURCE_SHA256,
                      V23_ASSESSMENT, digest, load_json, verify_frozen)


OUTPUT = HERE / "source_old_nine_fixed"


def main():
    if sys.flags.optimize:
        raise RuntimeError("Fixed-case assertions must remain enabled")
    verify_frozen()
    if load_json(HERE / "QUALIFIED_V25.json").get("state") != "verified_not_training":
        raise RuntimeError("Independent V25 zero-update qualification is required first")
    if OUTPUT.exists():
        raise FileExistsError("Original nine fixed-case assessment is one-shot")
    subprocess.run([sys.executable, "-u", str(V23_ASSESSMENT / "run_assessment.py"),
                    "--output", str(OUTPUT), "--sources",
                    f"source_v21:{SOURCE_COMPLETED}:{SOURCE}"], check=True)
    if digest(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("V21 source checkpoint changed during old-nine diagnostics")
    final = load_json(OUTPUT / "FINALIZED.json")
    if (final.get("state") != "complete" or final.get("optimizer_updates") != 0
            or final.get("hashes_unchanged") is not True
            or final.get("policy_count") != 2 or final.get("smoke_only") is not False):
        raise RuntimeError("Original-nine source fixed-case diagnostics are incomplete")


if __name__ == "__main__":
    main()
