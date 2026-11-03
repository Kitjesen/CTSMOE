"""CPU-readable actuator contract embedded in exported TorchScript policies."""

import json

import torch

from actuator_guard import LIMITS, PROFILE


CONTRACT_FILE = "actuator_contract.json"


def export_contract(profile):
    """Return the exact actuator contract expected by this runtime version."""
    if profile not in ("original", PROFILE):
        raise ValueError(f"Unknown actuator profile: {profile!r}")
    contract = {"schema_version": 1, "actuator_profile": profile}
    if profile == PROFILE:
        contract["limits"] = dict(LIMITS)
    return contract


def load_student_export(path):
    """Load a CPU evaluation policy and validate its embedded actuator contract.

    Older exports with no contract are explicitly classified as ``original``.
    Present but malformed, unsupported, or mismatched metadata is rejected;
    a guarded policy must never silently fall back to the original actuator setup.
    """
    extra_files = {CONTRACT_FILE: ""}
    policy = torch.jit.load(path, map_location="cpu", _extra_files=extra_files)
    payload = extra_files[CONTRACT_FILE]
    if not payload:
        contract = export_contract("original")
    else:
        try:
            contract = json.loads(payload)
        except (ValueError, UnicodeError) as error:
            raise ValueError("Invalid actuator contract JSON") from error
        if not isinstance(contract, dict):
            raise ValueError("Actuator contract must be a JSON object")
        if type(contract.get("schema_version")) is not int or contract["schema_version"] != 1:
            raise ValueError(f"Unsupported actuator contract schema: {contract.get('schema_version')!r}")
        expected = export_contract(contract.get("actuator_profile"))
        if contract != expected:
            raise ValueError("Actuator contract does not match the current profile and limits")
        contract = expected
    policy.eval()
    return policy, contract
