import torch

from pure_start_sampler import KIND_NAMES, explicit_selection, reset_selection, sample_explicit_goals


def test_shape_kind_and_exact_zero_structure():
    goals, kinds = sample_explicit_goals(2000, generator=torch.Generator().manual_seed(7))
    assert goals.shape == (2000, 3)
    assert kinds.shape == (2000,)
    assert torch.all((kinds >= 0) & (kinds < len(KIND_NAMES)))
    for kind in range(len(KIND_NAMES)):
        row = goals[kinds == kind]
        if kind in (0, 1, 6, 7, 8):
            assert torch.all((row[:, 0].abs() >= 0.2) & (row[:, 0].abs() <= 1.2))
        if kind in (2, 3, 6, 7, 9):
            assert torch.all((row[:, 1].abs() >= 0.2) & (row[:, 1].abs() <= 0.5))
        if kind in (4, 5, 8, 9):
            assert torch.all((row[:, 2].abs() >= 0.3) & (row[:, 2].abs() <= 0.8))


def test_each_kind_has_only_declared_components_and_fixed_signs():
    goals, kinds = sample_explicit_goals(10000, generator=torch.Generator().manual_seed(11))
    allowed = ((1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0),
               (0, 0, 1), (0, 0, 1), (1, 1, 0), (1, 1, 0),
               (1, 0, 1), (0, 1, 1))
    for kind, mask in enumerate(allowed):
        row = goals[kinds == kind]
        assert torch.all(row.ne(0) == torch.tensor(mask, dtype=torch.bool))
        if kind == 0:
            assert torch.all(row[:, 0] > 0)
        if kind == 1:
            assert torch.all(row[:, 0] < 0)
        if kind == 2:
            assert torch.all(row[:, 1] > 0)
        if kind == 3:
            assert torch.all(row[:, 1] < 0)
        if kind == 4:
            assert torch.all(row[:, 2] > 0)
        if kind == 5:
            assert torch.all(row[:, 2] < 0)


def test_two_cohorts_and_single_reset_inherit_global_mask():
    a = explicit_selection(4096, generator=torch.Generator().manual_seed(3))
    b = explicit_selection(4096, generator=torch.Generator().manual_seed(3))
    assert int(a.sum()) == 2048
    assert torch.equal(a, b)
    global_mask = torch.cat((a, b))
    chosen = global_mask.nonzero().flatten()[0:1]
    rejected = (~global_mask).nonzero().flatten()[0:1]
    assert torch.equal(reset_selection(chosen, global_mask), chosen)
    assert reset_selection(rejected, global_mask).numel() == 0
    assert torch.equal(reset_selection(torch.cat((chosen, rejected)), global_mask), chosen)


def test_empty_and_odd_selection():
    assert sample_explicit_goals(0)[0].shape == (0, 3)
    assert int(explicit_selection(5).sum()) == 2
