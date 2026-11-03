import json
from pathlib import Path

import pytest

import run_pipeline as pipeline


def test_training_budget_and_exact_completed_updates():
    assert pipeline.BASE_COMPLETED == 44711
    assert [item[1] for item in pipeline.TRAINING_STAGES] == [16, 284, 700]
    assert sum(item[1] for item in pipeline.TRAINING_STAGES) == 1000
    assert [item[2] for item in pipeline.TRAINING_STAGES] == [44727, 45011, 45711]


def test_first_train_imports_only_fixed_donor_coefficients_and_all_stages_use_reference():
    destination = pipeline.ROOT/'runs/example'
    first = pipeline.train_command(destination, pipeline.SOURCE, 16, first_stage=True)
    resumed = pipeline.train_command(destination, pipeline.SOURCE, 284, first_stage=False)
    assert first[first.index('--resume')+1] == str(pipeline.SOURCE)
    assert first[first.index('--startup-anchor-coefficients-from')+1] == str(pipeline.DONOR)
    assert '--startup-anchor-coefficients-from' not in resumed
    for command in (first, resumed):
        assert command[command.index('--startup-repair-profile')+1] == 'omni_full_episode_v1'
        assert command[command.index('--startup-anchor-reference')+1] == str(pipeline.REFERENCE)
        assert str(pipeline.DONOR) not in command[:command.index('--startup-anchor-reference')]


def test_evaluation_selections_have_exact_case_budgets_and_v6_only_on_flat25():
    checkpoint = pipeline.ROOT/'runs/candidate/model_45010.pt'
    normal = pipeline.selection(checkpoint, 45011)
    long = pipeline.selection(checkpoint, 45011, flat25=True)
    assert len(normal)*83 == 332
    assert len(long)*20 == 100
    assert all(item['label'] != 'V6_43202' for item in normal)
    assert sum(item['label'] == 'V6_43202' for item in long) == 1
    assert [item['label'] for item in normal] == [
        'V9_44711', 'V9_44711_teacher', 'V12_45011', 'V12_45011_teacher']


def test_all_four_gates_are_required_and_receive_final_flag():
    calls = []

    def normal_gate(analysis, baseline, candidate, final=False):
        calls.append(('normal', baseline, candidate, final))
        return {'accepted': candidate.endswith('_teacher')}

    def flat_gate(analysis, baseline, candidate, final=False):
        calls.append(('flat25', baseline, candidate, final))
        return {'accepted': True}

    result = pipeline.evaluate_gates({}, {}, 45711, normal_gate, flat_gate, final=True)
    assert len(calls) == 4
    assert all(call[-1] is True for call in calls)
    assert result['normal_student']['accepted'] is False
    assert result['accepted'] is False


def test_preconditions_require_finalized_zero_optimizer_preflight_and_verified_sources():
    valid = {'state': 'full_episode_preflight_complete_optimizer_steps_0',
             'functional_sampling_valid': True, 'optimizer_updates': 0,
             'completed_updates': 44711, 'control_steps': 3000, 'num_envs': 4096,
             'checkpoint_sha256': 'source', 'reference_sha256': 'reference',
             'coefficient_donor_sha256': 'donor'}
    artifacts = {'checkpoint_sha256': 'source', 'reference_sha256': 'reference',
                 'coefficient_donor_sha256': 'donor'}
    pipeline.validate_preflight(valid, artifacts)
    for replacement in ({'functional_sampling_valid': False}, {'optimizer_updates': 1},
                        {'state': 'incomplete'}):
        with pytest.raises(AssertionError):
            pipeline.validate_preflight({**valid, **replacement}, artifacts)
    with pytest.raises(AssertionError):
        pipeline.validate_preflight(valid, {**artifacts, 'checkpoint_sha256': 'replaced'})
    hashes = {'train.py': 'a'*64}
    pipeline.validate_verification(hashes, {'passed': True, 'source_hashes': hashes})
    with pytest.raises(AssertionError):
        pipeline.validate_verification(hashes, {'passed': False})


def test_pipeline_refuses_existing_outputs_and_persists_rejection(monkeypatch, tmp_path):
    existing = tmp_path/'existing'
    existing.mkdir()
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'STAGE', tmp_path)
    monkeypatch.setattr(pipeline, 'CODE', tmp_path)
    monkeypatch.setattr(pipeline, 'SOURCE', tmp_path/'source.pt')
    monkeypatch.setattr(pipeline, 'REFERENCE', tmp_path/'reference.pt')
    monkeypatch.setattr(pipeline, 'DONOR', tmp_path/'donor.pt')
    try:
        pipeline.write_json_new(existing/'evidence.json', {'state': 'first'})
        pipeline.write_json_new(existing/'evidence.json', {'state': 'second'})
    except FileExistsError:
        pass
    else:
        raise AssertionError('evidence overwrite was not refused')
    assert json.loads((existing/'evidence.json').read_text()) == {'state': 'first'}


def test_launcher_uses_global_lock_gpu_refusal_and_pinned_isaac_python():
    launcher = Path(pipeline.__file__).with_name('run_pipeline.sh').read_text(encoding='utf-8')
    assert 'flock -n 9' in launcher
    assert 'waiting_for_preflight' in launcher
    assert 'SECONDS + 3600' in launcher
    assert 'AUDIT_FAILED' in launcher
    assert 'pgrep -af' in launcher
    assert 'sleep 20' in launcher
    assert 'nvidia-smi --query-compute-apps=pid' in launcher
    assert '/root/autodl-tmp/conda/envs/isaaclab/bin/python' in launcher
