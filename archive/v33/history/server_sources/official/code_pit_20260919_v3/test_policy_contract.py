import json

import pytest
import torch

from actuator_guard import LIMITS, PEAK_LIMITS, PEAK_PROFILE, PROFILE, SLEW_PROFILE
from policy_contract import CONTRACT_FILE, checkpoint_profile, export_contract, load_student_export


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.register_buffer("offset", torch.tensor([0.1, -0.2]))

    def forward(self, observations):
        return self.linear(observations) + self.offset


@pytest.fixture
def traced_policy():
    return torch.jit.trace(TinyPolicy().train(), torch.ones(1, 3))


def save_export(tmp_path, traced_policy, payload=None):
    path = tmp_path / "student.pt"
    extra_files = {} if payload is None else {CONTRACT_FILE: payload}
    torch.jit.save(traced_policy, path, _extra_files=extra_files)
    return path


def test_guarded_export_roundtrips_actual_torchscript_with_cpu_eval(tmp_path, traced_policy):
    contract = export_contract(PROFILE)
    path = save_export(tmp_path, traced_policy, json.dumps(contract))
    loaded, actual = load_student_export(path)
    assert isinstance(loaded, torch.jit.ScriptModule)
    assert actual == {"schema_version": 1, "actuator_profile": PROFILE, "limits": LIMITS}
    assert not loaded.training
    assert all(not module.training for module in loaded.modules())
    assert all(tensor.device.type == "cpu" for tensor in (*loaded.parameters(), *loaded.buffers()))
    inputs = torch.tensor([[0.3, -0.8, 1.2], [-1.0, 0.0, 0.4]])
    torch.testing.assert_close(loaded(inputs), traced_policy(inputs))


@pytest.mark.parametrize("payload", [None, "", json.dumps(export_contract("original"))])
def test_legacy_or_explicit_original_policy_uses_original_contract(tmp_path, traced_policy, payload):
    path = save_export(tmp_path, traced_policy, payload)
    loaded, contract = load_student_export(path)
    assert contract == {"schema_version": 1, "actuator_profile": "original"}
    assert not loaded.training
    torch.testing.assert_close(loaded(torch.ones(1, 3)), traced_policy(torch.ones(1, 3)))


def test_export_contract_limits_are_an_independent_json_serializable_copy():
    contract = export_contract(PROFILE)
    assert json.loads(json.dumps(contract)) == contract
    contract["limits"]["leg_torque_nm"] = -1.0
    assert LIMITS["leg_torque_nm"] > 0
    assert export_contract(PROFILE)["limits"] == LIMITS


@pytest.mark.parametrize("profile", ["unknown", "", None, 1])
def test_unknown_export_profile_is_rejected(profile):
    with pytest.raises(ValueError, match="Unknown actuator profile"):
        export_contract(profile)


@pytest.mark.parametrize("schema", [2, 0, "1", True, 1.0, None])
def test_present_unsupported_schema_fails_closed(tmp_path, traced_policy, schema):
    contract = export_contract(PROFILE)
    contract["schema_version"] = schema
    path = save_export(tmp_path, traced_policy, json.dumps(contract))
    with pytest.raises(ValueError, match="Unsupported actuator contract schema"):
        load_student_export(path)


def test_present_unknown_profile_fails_closed(tmp_path, traced_policy):
    contract = {"schema_version": 1, "actuator_profile": "unrecognized_guard"}
    path = save_export(tmp_path, traced_policy, json.dumps(contract))
    with pytest.raises(ValueError, match="Unknown actuator profile"):
        load_student_export(path)


@pytest.mark.parametrize("mutation", ["changed_limit", "missing_limit", "missing_limits", "extra_field"])
def test_incompatible_guard_limits_or_fields_fail_closed(tmp_path, traced_policy, mutation):
    contract = export_contract(PROFILE)
    if mutation == "changed_limit":
        contract["limits"]["leg_torque_nm"] += 1.0
    elif mutation == "missing_limit":
        del contract["limits"]["wheel_speed_rad_s"]
    elif mutation == "missing_limits":
        del contract["limits"]
    else:
        contract["unrecognized_field"] = "do not silently ignore"
    path = save_export(tmp_path, traced_policy, json.dumps(contract))
    with pytest.raises(ValueError, match="does not match"):
        load_student_export(path)


@pytest.mark.parametrize("payload", ["not json", " ", "[]", "null", "{}"])
def test_present_malformed_metadata_never_falls_back_to_legacy(tmp_path, traced_policy, payload):
    path = save_export(tmp_path, traced_policy, payload)
    with pytest.raises(ValueError):
        load_student_export(path)


def test_original_metadata_cannot_smuggle_guard_limits(tmp_path, traced_policy):
    contract = export_contract("original")
    contract["limits"] = dict(LIMITS)
    path = save_export(tmp_path, traced_policy, json.dumps(contract))
    with pytest.raises(ValueError, match="does not match"):
        load_student_export(path)


def test_v2_export_roundtrips_with_both_hard_limits_and_soft_references(tmp_path, traced_policy):
    contract = export_contract(PEAK_PROFILE)
    assert contract == {"schema_version": 1, "actuator_profile": PEAK_PROFILE, "limits": PEAK_LIMITS}
    assert contract["limits"]["leg_torque_nm"] == 120.0
    assert contract["limits"]["wheel_torque_nm"] == 17.0
    assert contract["limits"]["leg_load_reference_nm"] == 35.0
    assert contract["limits"]["wheel_load_reference_nm"] == 6.0
    path = save_export(tmp_path, traced_policy, json.dumps(contract))
    loaded, actual = load_student_export(path)
    assert actual == contract
    assert not loaded.training
    assert all(tensor.device.type == "cpu" for tensor in (*loaded.parameters(), *loaded.buffers()))
    torch.testing.assert_close(loaded(torch.ones(2, 3)), traced_policy(torch.ones(2, 3)))


@pytest.mark.parametrize("field", ["leg_torque_nm", "wheel_torque_nm", "leg_load_reference_nm",
                                  "wheel_load_reference_nm", "leg_speed_rad_s"])
def test_v2_jit_export_rejects_changed_hard_or_soft_contract_values(tmp_path, traced_policy, field):
    contract = export_contract(PEAK_PROFILE)
    contract["limits"][field] += 1.0
    path = save_export(tmp_path, traced_policy, json.dumps(contract))
    with pytest.raises(ValueError, match="does not match"):
        load_student_export(path)


@pytest.mark.parametrize("profile", ["original", PROFILE])
def test_legacy_checkpoints_remain_accepted_without_embedded_contract(profile):
    assert checkpoint_profile({"actuator_profile": profile}) == profile
    assert checkpoint_profile({}) == "original"


@pytest.mark.parametrize("profile", ["original", PROFILE, PEAK_PROFILE, SLEW_PROFILE])
def test_checkpoint_accepts_matching_complete_contract(profile):
    checkpoint = {"actuator_profile": profile, "actuator_contract": export_contract(profile)}
    assert checkpoint_profile(checkpoint) == profile


@pytest.mark.parametrize("contract", [None, {}, export_contract(PROFILE), export_contract("original")])
def test_v2_checkpoint_rejects_absent_incomplete_or_other_profiles_contract(contract):
    checkpoint = {"actuator_profile": PEAK_PROFILE, "actuator_contract": contract}
    with pytest.raises(ValueError, match="missing or mismatched"):
        checkpoint_profile(checkpoint)
    with pytest.raises(ValueError, match="missing or mismatched"):
        checkpoint_profile({"actuator_profile": PEAK_PROFILE})


@pytest.mark.parametrize("mutation", ["hard_limit", "soft_reference", "missing_soft_reference",
                                     "missing_limits", "schema", "profile", "extra_field"])
def test_v2_checkpoint_rejects_inconsistent_complete_contract(mutation):
    contract = export_contract(PEAK_PROFILE)
    if mutation == "hard_limit":
        contract["limits"]["leg_torque_nm"] = 35.0
    elif mutation == "soft_reference":
        contract["limits"]["leg_load_reference_nm"] = 120.0
    elif mutation == "missing_soft_reference":
        del contract["limits"]["wheel_load_reference_nm"]
    elif mutation == "missing_limits":
        del contract["limits"]
    elif mutation == "schema":
        contract["schema_version"] = 2
    elif mutation == "profile":
        contract["actuator_profile"] = PROFILE
    else:
        contract["extra_field"] = True
    with pytest.raises(ValueError, match="missing or mismatched"):
        checkpoint_profile({"actuator_profile": PEAK_PROFILE, "actuator_contract": contract})


@pytest.mark.parametrize("schema", [True, 1.0])
def test_checkpoint_schema_must_remain_an_integer_version(schema):
    contract = export_contract(PEAK_PROFILE)
    contract["schema_version"] = schema
    with pytest.raises(ValueError):
        checkpoint_profile({"actuator_profile": PEAK_PROFILE, "actuator_contract": contract})


def test_checkpoint_rejects_unknown_profile_and_contract_without_matching_top_level_profile():
    with pytest.raises(ValueError, match="Unknown actuator profile"):
        checkpoint_profile({"actuator_profile": "unknown"})
    with pytest.raises(ValueError, match="missing or mismatched"):
        checkpoint_profile({"actuator_contract": export_contract(PEAK_PROFILE)})


def test_slew12_export_and_checkpoint_roundtrip_without_reinterpreting_v2(tmp_path, traced_policy):
    contract = export_contract(SLEW_PROFILE)
    assert contract['limits'] == {**PEAK_LIMITS, 'leg_target_rate_rad_s': 12.0}
    assert export_contract(PEAK_PROFILE)['limits']['leg_target_rate_rad_s'] == 6.0
    loaded, actual = load_student_export(save_export(tmp_path, traced_policy, json.dumps(contract)))
    assert actual == contract
    torch.testing.assert_close(loaded(torch.ones(2, 3)), traced_policy(torch.ones(2, 3)))
    assert checkpoint_profile({'actuator_profile': SLEW_PROFILE, 'actuator_contract': contract}) == SLEW_PROFILE
    with pytest.raises(ValueError, match='missing or mismatched'):
        checkpoint_profile({'actuator_profile': SLEW_PROFILE})
    with pytest.raises(ValueError, match='missing or mismatched'):
        checkpoint_profile({'actuator_profile': SLEW_PROFILE, 'actuator_contract': export_contract(PEAK_PROFILE)})
    contract['limits']['leg_target_rate_rad_s'] = 6.0
    with pytest.raises(ValueError, match='does not match'):
        load_student_export(save_export(tmp_path, traced_policy, json.dumps(contract)))
