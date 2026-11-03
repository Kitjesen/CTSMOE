import prepare


def test_protocol_is_bounded_and_spans_registered_windows():
    manifest = prepare.protocol_manifest()
    assert manifest['control_steps'] == 288 <= 320
    assert manifest['native_horizon_steps'] == 24
    assert manifest['windows_s'] == {'startup_0_1': [0., 1.], 'sustained_1_5': [1., 5.]}
    assert 288/50 > 5.


def test_protocol_requires_real_on_policy_ppo_and_labels_counterfactual_anchor():
    truth = prepare.protocol_manifest()['truth_constraints']
    assert truth['fresh_on_policy_advantages_required']
    assert truth['compute_returns_required_each_native_horizon']
    assert truth['old_v11_offpolicy_observations_are_not_ppo_gradients']
    assert truth['post_1s_anchor_is_counterfactual']


def test_preparation_has_no_training_or_coefficient_mutation_authority():
    manifest = prepare.protocol_manifest()
    assert manifest['state'] == 'PREPARED_NOT_EXECUTED'
    assert manifest['optimizer_steps'] == 0
    assert manifest['checkpoint_writes'] == 0
    assert manifest['coefficient_changes'] == 0
    assert manifest['requires_parent_execution_decision']


def test_scarcity_and_conflict_have_separate_registered_evidence():
    manifest = prepare.protocol_manifest()
    assert 'raw rows and unique environments by window/teacher/student' in manifest['scarcity_evidence']
    assert 'pairwise cosine and dot product on exactly shared parameter vectors' in manifest['conflict_evidence']
    assert 'nonzero-advantage and nonzero-gradient counts' in manifest['scarcity_evidence']
