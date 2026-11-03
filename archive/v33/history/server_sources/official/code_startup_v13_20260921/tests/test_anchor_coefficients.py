import copy
import hashlib

import pytest
import torch

import anchor_coefficients as module


def donor_files(tmp_path, monkeypatch, coefficients=None):
    reference = tmp_path / 'reference.pt'
    reference.write_bytes(b'unit-test-reference')
    ref_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    state = {'completed_updates': 44759, 'model_state_dict': {'must_not_transfer': torch.ones(2)},
             'startup_anchor': {'profile': 'v6_normal_start_action_v1',
                'reference_sha256': ref_hash, 'coefficients': coefficients or copy.deepcopy(module.COEFFICIENTS),
                'calibration': {'main': {}, 'student': {}}}}
    path = tmp_path / 'donor.pt'
    torch.save(state, path)
    monkeypatch.setattr(module, 'SOURCE_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(module, 'REFERENCE_SHA256', ref_hash)
    return path, reference


def test_only_anchor_metadata_imported_and_source_recorded(tmp_path, monkeypatch):
    donor, reference = donor_files(tmp_path, monkeypatch)
    state = module.import_fixed_coefficients(donor, reference)
    assert 'model_state_dict' not in state
    assert state['coefficients'] == module.COEFFICIENTS
    assert state['coefficient_origin']['calibration_disabled']
    assert state['coefficient_origin']['donor_completed_updates'] == 44759


def test_modified_donor_rejected(tmp_path, monkeypatch):
    donor, reference = donor_files(tmp_path, monkeypatch)
    donor.write_bytes(donor.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='audited V11'):
        module.import_fixed_coefficients(donor, reference)


def test_different_coefficients_rejected(tmp_path, monkeypatch):
    donor, reference = donor_files(tmp_path, monkeypatch, {'main': .9, 'student': .01})
    with pytest.raises(ValueError, match='coefficient changed'):
        module.import_fixed_coefficients(donor, reference)
