"""Import the audited fixed anchor coefficients without importing policy weights."""
import copy
import hashlib
import math

import torch


SOURCE_SHA256 = "73d961f5855d36b43d86852845a31f9a27259805cea53a317787e0bf023e5c76"
REFERENCE_SHA256 = "bdde0ca02d7e0d9db9ecdfd8c9b352cc1f511db49a9af936cc14c8046f6a77f8"
COEFFICIENTS = {"main": 0.21093362834788681, "student": 0.0022221143221353456}


def import_fixed_coefficients(path, reference_path):
    """Return anchor metadata only, with an explicit donor provenance record."""
    path = path.resolve(strict=True)
    reference_path = reference_path.resolve(strict=True)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError("Coefficient donor must be the audited V11 44759 checkpoint")
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Coefficient import requires the audited frozen V6 reference")
    donor = torch.load(path, map_location="cpu", weights_only=False)
    state = copy.deepcopy(donor.get("startup_anchor"))
    if donor.get("completed_updates") != 44759 or not isinstance(state, dict):
        raise ValueError("Coefficient donor has invalid iteration or anchor metadata")
    if state.get("profile") != "v6_normal_start_action_v1" or state.get("reference_sha256") != REFERENCE_SHA256:
        raise ValueError("Coefficient donor does not describe the validated V6 anchor")
    values = state.get("coefficients")
    if not isinstance(values, dict) or set(values) != set(COEFFICIENTS):
        raise ValueError("Coefficient donor schema changed")
    for key, expected in COEFFICIENTS.items():
        actual = values[key]
        if type(actual) not in (float, int) or not math.isclose(actual, expected, rel_tol=0., abs_tol=1e-15):
            raise ValueError(f"Audited {key} coefficient changed")
    state["reference"] = str(reference_path)
    state["coefficient_origin"] = {
        "mode": "fixed_import_without_model_or_optimizer_transfer",
        "path": str(path),
        "sha256": source_hash,
        "donor_completed_updates": 44759,
        "calibration_disabled": True,
    }
    return state
