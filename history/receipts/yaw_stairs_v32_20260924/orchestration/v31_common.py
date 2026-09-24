"""Frozen, bounded V31 launch contract. No credentials are stored here."""

import hashlib
import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917")
STAGE = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
LOCK = Path("/root/autodl-tmp/cts-omni-repair-v19-20260922/full_episode_v12_pipeline.lock")
ENVS = 4096
STEPS = 3000


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def frozen():
    receipt = json.loads((HERE / "V31_FROZEN.json").read_text())
    actual = {p.name: digest(p) for p in (STAGE / "code").glob("*.py")}
    if actual != receipt["code"]:
        raise ValueError("V31 training source changed since freezing")
    scripts = {name: digest(HERE / name) for name in (
        "v31_common.py", "v31_preflight.py", "state_guard.py")}
    if scripts != receipt["scripts"]:
        raise ValueError("V31 orchestration changed since freezing")
    if digest(receipt["source_path"]) != receipt["source_sha256"]:
        raise ValueError("V21 source checkpoint changed")
    return receipt


def args(source, output, updates):
    return [
        "--terrain-variety", "--root", str(BASE), "--output", str(output), "--num-envs", str(ENVS),
        "--updates", str(updates), "--mixed-pit", "--resume", str(source),
        "--early-restart-profile", "early_phase_v14_no_reward",
        "--startup-repair-profile", "off", "--pit-posture-profile", "off",
        "--surface-posture-profile", "off", "--stance-strength-profile", "off",
        "--allocation-profile", "balanced_adaptive_v1",
        "--pit-training-profile", "strict_v1", "--pit-course-profile", "global_v1",
        "--no-pit-profile", "omni_eight_v1",
        "--hip-margin-profile", "actual_prehard_v1",
        "--hip-request-profile", "off",
        "--mixed-stair-profile", "mixed_stair_short_pulses_v1",
        "--actuator-profile", "rs0402_transient_v4",
        "--slope-profile", "slope45_v1", "--command-curriculum-start", "intermediate",
        "--base-height-penalty", "original", "--terrain-training-profile", "stairs80_v1",
        "--stair-command-profile", "straight_speed_v1",
        "--wave-reset-profile", "wave_surface_v1",
    ]
