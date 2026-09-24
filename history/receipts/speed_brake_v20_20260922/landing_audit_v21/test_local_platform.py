import pytest
import torch

from local_platform import local_platform_support, phase_ids, platform_from_probes, update_dwell


def inputs():
    return [torch.tensor([[[3., -.5, .4], [4., .5, 1.]]]),
            torch.zeros(1, 4), torch.zeros(1, 4), torch.ones(1, 4) * 10,
            torch.zeros(1)]


def test_support_uses_local_geometry_not_unrelated_scan():
    # A wide scanner may see a 34cm riser; it is intentionally not an input.
    assert local_platform_support(*inputs())['supported'].item()


def test_two_contacts_rejected():
    values = inputs()
    values[3][0, 2:] = 0
    assert not local_platform_support(*values)['supported'].item()


def test_wrong_platform_height_rejected():
    values = inputs()
    values[1] += .2
    assert not local_platform_support(*values)['supported'].item()


def test_three_valid_contacts_with_fourth_ray_missing():
    values = inputs()
    values[1][0, 3] = float('nan')
    assert local_platform_support(*values)['supported'].item()


def test_non_coplanar_support_rejected():
    values = inputs()
    values[1][0] = torch.tensor([-.029, .029, .029, float('nan')])
    assert not local_platform_support(*values)['supported'].item()


def test_full_hull_required():
    values = inputs()
    values[0][0, 0, 0] = 2.8
    assert not local_platform_support(*values)['supported'].item()


def test_exact_two_second_dwell_and_interruption():
    count = torch.zeros(1, dtype=torch.int64)
    for _ in range(99):
        count, done = update_dwell(count, torch.tensor([True]))
        assert not done.item()
    count, done = update_dwell(count, torch.tensor([True]))
    assert done.item() and count.item() == 100
    count, done = update_dwell(count, torch.tensor([False]))
    assert not done.item() and count.item() == 0


def test_up_and_down_known_platform_height():
    values = inputs()
    for height in (1.2, -1.2):
        values[1].fill_(height)
        values[4].fill_(height)
        assert local_platform_support(*values)['supported'].item()


def test_authored_platform_probes_must_be_finite_and_flat():
    assert platform_from_probes(torch.ones(2, 9)).tolist() == [1., 1.]
    bad = torch.zeros(1, 9)
    bad[0, 0] = .09
    with pytest.raises(ValueError, match='not flat'):
        platform_from_probes(bad)


def test_phases_are_disjoint_and_do_not_call_climbing_rebound():
    edge = torch.tensor([False, True, True, True, True])
    hull = torch.tensor([False, False, True, True, True])
    supported = torch.tensor([False, False, False, True, True])
    done = torch.tensor([False, False, False, False, True])
    ids = phase_ids(edge, hull, supported, done)
    assert ids.tolist() == list(range(5))
    vz = torch.tensor([.8, .7, .1, .03, .01])
    assert vz[ids >= 2].max().item() < .11
