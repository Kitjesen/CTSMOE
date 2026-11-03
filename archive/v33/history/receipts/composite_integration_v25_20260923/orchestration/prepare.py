"""CPU-only freeze of the sole reviewed V21 source and V25 code."""

from __future__ import annotations

import json

from contract import (HERE, SOURCE, SOURCE_COMPLETED, SOURCE_SHA256, STAGE,
                      STATE_KEYS, V21_FIXED_PROTOCOL, V23_ASSESSMENT,
                      budget, digest, snapshot, validate_source, write_new)
from state_guard import state_sha256


def prepare() -> dict:
    import torch

    target = HERE / "FROZEN_V25.json"
    if target.exists() or (HERE / "STARTED_V25.json").exists():
        raise FileExistsError("V25 has already been frozen or attempted")
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError("Reviewed V21 checkpoint SHA-256 differs")
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    validate_source(saved)
    hashes = {name: state_sha256(saved[key], torch) for name, key in STATE_KEYS.items()}
    frozen = dict(stage="composite_route_v25",
                  source=dict(path=str(SOURCE), sha256=SOURCE_SHA256,
                              completed_updates=SOURCE_COMPLETED),
                  source_state_sha256=hashes, candidate_code=snapshot(STAGE / "code"),
                  orchestration=snapshot(HERE),
                  v23_assessment=snapshot(V23_ASSESSMENT),
                  v21_fixed_protocol_sha256=digest(V21_FIXED_PROTOCOL),
                  budget=budget(0),
                  zero_update_control_steps=3000, automatic_training=False,
                  real_robot_release=False)
    write_new(target, frozen)
    return frozen


if __name__ == "__main__":
    print(json.dumps(prepare()["source"]))
