from types import SimpleNamespace

import torch

from tensordict import TensorDict

from ledger import (Accumulator, GradientLedger, SAMPLES_PER_HORIZON, STUDENT_SAMPLES, TEACHER_SAMPLES,
                    deterministic_subsample, pair_metrics, policy_branch_results)
from launcher import MutationGuard, augment_observations, first_metric


def test_pair_metrics_preserve_norm_dot_and_cosine():
    result = pair_metrics(torch.tensor([1., 0.]), torch.tensor([0., 2.]), .25)
    assert result['primary_gradient_norm'] == 1.
    assert result['anchor_gradient_norm'] == 2.
    assert result['weighted_anchor_gradient_norm'] == .5
    assert result['dot_product'] == 0.
    assert result['cosine'] == 0.


def test_first_metric_reads_training_configuration_without_loading_whole_receipt(tmp_path):
    receipt = tmp_path/'metrics.jsonl'
    receipt.write_text('{"fast_delay": false}\n{"fast_delay": true}\n', encoding='utf-8')
    assert first_metric(receipt) == {'fast_delay': False}


def test_accumulator_weights_window_gradients_by_registered_sample_count():
    accumulator = Accumulator()
    accumulator.add('window/main', torch.tensor([1.]), torch.tensor([2.]), .5, 1, 1)
    accumulator.add('window/main', torch.tensor([3.]), torch.tensor([4.]), .5, 3, 2)
    result = accumulator.finish()['window/main']
    assert result['primary_gradient_norm'] == 2.5
    assert result['anchor_gradient_norm'] == 3.5
    assert result['aggregation_weight'] == 4
    assert result['conflict_evidence_rows'] == 3


def test_deterministic_subsample_keeps_three_to_one_cts_partition():
    alg = SimpleNamespace(
        teacher_num_envs=3072, student_num_envs=1024, num_mini_batches=4,
        storage=SimpleNamespace(num_transitions_per_env=24))
    teacher, student = 3072*24//4, 1024*24//4
    values = torch.arange(teacher+student)
    batch = (values, values, values, values, values, values, values, values, None, None)
    selected, native_teacher, native_student = deterministic_subsample(batch, alg)
    assert len(selected[0]) == SAMPLES_PER_HORIZON
    assert (selected[0][:TEACHER_SAMPLES] < teacher).all()
    assert (selected[0][TEACHER_SAMPLES:] >= teacher).all()
    assert len(selected[0][TEACHER_SAMPLES:]) == STUDENT_SAMPLES
    assert (native_teacher, native_student) == (teacher, student)


def test_mutation_guard_blocks_updates_but_deliberately_allows_compute_returns():
    calls = []
    optimizer = SimpleNamespace(step=lambda: calls.append('step'))
    student = SimpleNamespace(step=lambda: calls.append('student_step'))
    alg = SimpleNamespace(update=lambda: calls.append('update'),
                          compute_returns=lambda obs: calls.append(('returns', obs)),
                          optimizer=optimizer, optimizer_stu_enc=student)
    runner = SimpleNamespace(alg=alg, save=lambda: calls.append('save'))
    guard = MutationGuard(runner)
    with guard:
        alg.compute_returns('fresh_on_policy')
        try:
            alg.update()
        except RuntimeError:
            pass
    assert calls == [('returns', 'fresh_on_policy')]
    assert guard.calls == ['alg.update']
    assert guard.restored


def test_policy_branch_receives_observations_not_slice_object():
    class Policy:
        entropy = torch.ones(2)
        action_mean = torch.zeros(2, 16)

        def act(self, obs, teacher):
            assert torch.is_tensor(obs)
            assert obs.shape == (2, 3)

        def get_actions_log_prob(self, actions):
            return torch.zeros(len(actions))

    obs, actions = torch.zeros(4, 3), torch.zeros(4, 16)
    result = policy_branch_results(Policy(), obs, actions, slice(0, 2), True)
    assert len(result[0]) == 2


def test_telemetry_uses_position_request_and_wheel_velocity_target_without_wheel_request():
    count = 2
    obs = TensorDict({'policy': torch.zeros(count, 1)}, batch_size=[count])
    position = SimpleNamespace(joint_indices=list(range(12)),
                               requested_actions=torch.ones(count, 12))
    # Match native IsaacLab JointVelocityAction: no custom joint_indices property.
    wheels = SimpleNamespace(_joint_ids=list(range(12, 16)))
    action_manager = SimpleNamespace(
        get_term=lambda name: position if name == 'joint_pos' else wheels)
    robot_data = SimpleNamespace(
        joint_pos=torch.zeros(count, 16), joint_pos_target=torch.ones(count, 16)*2,
        joint_vel=torch.ones(count, 16)*3, joint_vel_target=torch.ones(count, 16)*4)
    contacts = [SimpleNamespace(data=SimpleNamespace(
        force_matrix_w=torch.ones(count, 1, 1, 3)*6)) for _ in range(4)]
    pit = SimpleNamespace(
        ground_contacts=contacts, base_contact_ids=[0],
        contact=SimpleNamespace(data=SimpleNamespace(
            net_forces_w_history=torch.zeros(count, 1, 1, 3))))
    command = SimpleNamespace(
        pit=pit, command=torch.zeros(count, 3), terrain_idxs=torch.zeros(count, dtype=torch.long),
        terrain_type2idx={'flat': 0})
    raw = SimpleNamespace(
        num_envs=count, device='cpu', episode_length_buf=torch.zeros(count), step_dt=.02,
        command_manager=SimpleNamespace(get_term=lambda name: command), action_manager=action_manager,
        scene={'robot': SimpleNamespace(data=robot_data)})
    alg = SimpleNamespace(student_env_idxs=torch.tensor([1]), anchor_cohort=torch.tensor([True, True]))

    result = augment_observations(obs, alg, raw, torch)

    assert result['audit_position_request'].shape == (count, 12)
    assert result['audit_wheel_velocity_target'].shape == (count, 4)
    torch.testing.assert_close(result['audit_wheel_velocity'], torch.ones(count, 4)*3)


def test_ledger_uses_native_joint_velocity_private_ids_contract(tmp_path):
    actor, teacher, student = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    policy = SimpleNamespace(actor=actor, teacher_encoder=teacher, student_moe_encoder=student)
    alg = SimpleNamespace(
        policy=policy,
        optimizer=torch.optim.SGD((*actor.parameters(), *teacher.parameters()), lr=.1),
        optimizer_stu_enc=torch.optim.SGD(student.parameters(), lr=.1),
        anchor_coefficients={'main': 0.21093362834788681,
                             'student': 0.0022221143221353456},
    )
    position = SimpleNamespace(joint_indices=list(range(12)))
    wheels = SimpleNamespace(_joint_ids=list(range(12, 16)))
    action_manager = SimpleNamespace(
        active_terms=['joint_pos', 'joint_vel'],
        get_term=lambda name: position if name == 'joint_pos' else wheels,
    )
    raw = SimpleNamespace(
        action_manager=action_manager,
        scene={'robot': SimpleNamespace(joint_names=[f'joint_{index}' for index in range(16)])},
    )

    ledger = GradientLedger(alg, raw, tmp_path)

    assert ledger.action_channels['wheel_joint_ids'] == list(range(12, 16))
