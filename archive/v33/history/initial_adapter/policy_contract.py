"""CPU-readable actuator contract embedded in exported TorchScript policies."""

import json

import torch

from actuator_guard import PEAK_PROFILE, PROFILE, SLEW_PROFILE, TRANSIENT_PROFILE, limits_for_profile


CONTRACT_FILE = "actuator_contract.json"


def export_contract(profile):
    """Return the exact actuator contract expected by this runtime version."""
    if profile not in ("original", PROFILE, PEAK_PROFILE, SLEW_PROFILE, TRANSIENT_PROFILE):
        raise ValueError(f"Unknown actuator profile: {profile!r}")
    contract = {"schema_version": 1, "actuator_profile": profile}
    if profile != "original":
        contract["limits"] = limits_for_profile(profile)
    return contract


def checkpoint_profile(checkpoint):
    """V2 and newer checkpoints bind complete limits; legacy formats remain readable."""
    profile = checkpoint.get("actuator_profile", "original")
    expected = export_contract(profile)
    actual = checkpoint.get("actuator_contract")
    if profile in (PEAK_PROFILE, SLEW_PROFILE, TRANSIENT_PROFILE) or actual is not None:
        if (not isinstance(actual, dict)
                or type(actual.get("schema_version")) is not int
                or actual != expected):
            raise ValueError("Checkpoint actuator contract is missing or mismatched")
    return profile


def load_student_export(path, *, expected_pose="asset_original"):
    """Load a CPU evaluation policy and validate its embedded actuator contract.

    Older exports with no contract are explicitly classified as ``original``.
    Present but malformed, unsupported, or mismatched metadata is rejected;
    a guarded policy must never silently fall back to the original actuator setup.
    """
    from default_pose import EXPORT_FILE, contract as pose_contract
    extra_files = {CONTRACT_FILE: "", EXPORT_FILE: ""}
    policy = torch.jit.load(path, map_location="cpu", _extra_files=extra_files)
    pose_payload = extra_files[EXPORT_FILE]
    actual_pose = json.loads(pose_payload) if pose_payload else pose_contract("asset_original")
    if actual_pose != pose_contract(expected_pose):
        raise ValueError("Export default pose mismatch; configure the matching pose before evaluation")
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
