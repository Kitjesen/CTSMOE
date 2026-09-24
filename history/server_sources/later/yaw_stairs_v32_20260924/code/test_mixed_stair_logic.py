"""Risk-focused offline tests; no simulator or frozen V23 code is imported."""

from types import SimpleNamespace

import pytest
import torch

from mixed_stair_logic import (
    LATERAL, STRAIGHT, YAW, MixedStairCourse, PlatformWitness, StairGeometry,
    actual_pose_in_stair_frame, conservative_center_hull_support,
    curriculum_decision, integrate_body_twist,
    integrate_schedule, planned_route_is_in_lane, sample_cohorts,
    scheduled_command, signed_path_error,
)
from mixed_stair_runtime import MixedStairCommandMixin, checkpoint_payload, restore_checkpoint


@pytest.mark.parametrize("group", [YAW, LATERAL])
def test_ignoring_both_mixed_command_pulses_cannot_promote(group):
    course = MixedStairCourse(1)
    ids = torch.tensor([0])
    course.start(ids, torch.tensor([group]), torch.tensor([1]),
                 torch.tensor([2]), torch.tensor([.6]))
    for step in range(225):
        course.elapsed_s[0] = step * .02
        issued = scheduled_command(course.group[ids], course.sign[ids],
                                   course.speed[ids], course.elapsed_s[ids])
        course.observe_tracking(ids, torch.tensor([[.6, 0., 0.]]), issued)
    assert not bool(course.pulse_followed(ids)[0])
    assert bool((course.pulse_count[0] >= 20).all())
    course.start(ids, torch.tensor([group]), torch.tensor([1]),
                 torch.tensor([2]), torch.tensor([.6]))
    for step in range(225):
        course.elapsed_s[0] = step * .02
        issued = scheduled_command(course.group[ids], course.sign[ids],
                                   course.speed[ids], course.elapsed_s[ids])
        course.observe_tracking(ids, issued, issued)
    assert bool(course.pulse_followed(ids)[0])


def terminal_receipt(kind="stairs_up", selected=True):
    return {"selected": selected, "kind": kind, "row": 9,
            "column": 8 if kind == "stairs_up" else 11,
            "geometry_after": {axis: {"surface_runs": [
                {"from_m": 0.0, "to_m": 1.1, "z_m": 0.0},
                {"from_m": 1.3, "to_m": 2.7, "z_m": 0.25},
                {"from_m": 2.8, "to_m": 4.5, "z_m": 0.50},
            ]} for axis in ("positive_x", "positive_y", "negative_x", "negative_y")}}


def test_sampling_retains_straight_and_both_signed_mixed_groups():
    groups, signs, bins, speeds = sample_cohorts(10_000, "cpu", torch.Generator().manual_seed(7))
    counts = torch.bincount(groups, minlength=3)
    assert 7700 < counts[STRAIGHT] < 8300
    assert 800 < counts[YAW] < 1200
    assert 800 < counts[LATERAL] < 1200
    assert set(signs[groups == STRAIGHT].tolist()) == {0}
    assert set(signs[groups == YAW].tolist()) == {-1, 1}
    assert set(signs[groups == LATERAL].tolist()) == {-1, 1}
    assert bool(((speeds[groups != STRAIGHT] >= 0.4) & (speeds[groups != STRAIGHT] < 0.8)).all())
    assert bool(((bins >= 0) & (bins < 4)).all())


@pytest.mark.parametrize("group", (YAW, LATERAL))
@pytest.mark.parametrize("sign", (-1, 1))
def test_signed_pulse_survives_updates_and_commanded_path_stays_in_lane(group, sign):
    groups, signs, speeds = torch.tensor([group]), torch.tensor([sign]), torch.tensor([0.6])
    times = torch.tensor([0., 2.2, 3.2, 4.1])
    commands = [scheduled_command(groups, signs, speeds, time.view(1))[0] for time in times]
    assert commands[0].tolist() == [pytest.approx(0.6), 0., 0.]
    channel = 2 if group == YAW else 1
    assert commands[1][channel].item() * sign > 0
    assert commands[2][channel].item() * sign < 0
    assert commands[3][channel].item() == 0
    geometry = StairGeometry.from_coverage_receipt(terminal_receipt(), "positive_x")
    assert planned_route_is_in_lane(geometry, group, sign, 0.6)
    assert not planned_route_is_in_lane(geometry, group, sign, 0.2)


def test_exact_turn_integral_and_split_at_pulse_boundary():
    pose = torch.zeros((1, 3), dtype=torch.float64)
    command = torch.tensor([[1., 0., torch.pi / 2]], dtype=torch.float64)
    result = integrate_body_twist(pose, command, torch.tensor([1.], dtype=torch.float64))
    assert result[0].tolist() == pytest.approx([2 / torch.pi, 2 / torch.pi, torch.pi / 2], abs=1e-8)
    group, sign, speed = torch.tensor([YAW]), torch.tensor([1]), torch.tensor([0.6], dtype=torch.float64)
    wide = integrate_schedule(pose, group, sign, speed, torch.tensor([1.98], dtype=torch.float64), .04)
    first = integrate_schedule(pose, group, sign, speed, torch.tensor([1.98], dtype=torch.float64), .02)
    split = integrate_schedule(first, group, sign, speed, torch.tensor([2.0], dtype=torch.float64), .02)
    torch.testing.assert_close(wide, split, rtol=0, atol=1e-10)


def test_platform_requires_four_real_terrain_contacts_for_consecutive_steps():
    geometry = StairGeometry.from_coverage_receipt(terminal_receipt(), "positive_x")
    witness = PlatformWitness(geometry, required_steps=10)
    points = torch.tensor([[3.0, -0.2], [3.0, 0.2], [3.4, -0.2], [3.4, 0.2]])
    forces = torch.full((4,), 20.)
    assert not witness.observe(points, torch.tensor([20., 20., 20., 0.]))
    for _ in range(9):
        assert not witness.observe(points, forces)
    assert witness.observe(points, forces)
    assert not PlatformWitness(geometry).observe(points, forces, physical_failure=True)
    with pytest.raises(ValueError, match="four"):
        witness.observe(points[:3], forces)
    bad = terminal_receipt(selected=False)
    with pytest.raises(ValueError, match="selected"):
        StairGeometry.from_coverage_receipt(bad, "positive_x")


def test_conservative_witness_requires_inset_wheel_centers_body_hull_and_four_forces():
    wheel_centers = torch.tensor([[[3.15, -.30], [3.15, .30], [3.45, -.30], [3.45, .30]]])
    hull = torch.tensor([[[3.05, -.40], [3.05, .40], [3.60, -.40], [3.60, .40]]])
    forces = torch.full((1, 4), 20.)
    args = (torch.full((1, 4), .62), torch.tensor([.5]), forces,
            torch.tensor([2.8]), torch.tensor([4.5]), torch.tensor([1.0]), .12)
    assert conservative_center_hull_support(wheel_centers, hull, *args).tolist() == [True]
    unsafe_wheel = wheel_centers.clone()
    unsafe_wheel[0, 0, 0] = 2.95  # Inside polygon, but its collision radius can reach the riser.
    assert conservative_center_hull_support(unsafe_wheel, hull, *args).tolist() == [False]
    unsafe_hull = hull.clone()
    unsafe_hull[0, 0, 0] = 2.85
    assert conservative_center_hull_support(wheel_centers, unsafe_hull, *args).tolist() == [False]
    unsupported = (*args[:2], torch.tensor([[20., 20., 20., 0.]]), *args[3:])
    assert conservative_center_hull_support(wheel_centers, hull, *unsupported).tolist() == [False]
    lower_surface = (args[0], torch.tensor([.25]), *args[2:])
    assert conservative_center_hull_support(wheel_centers, hull, *lower_surface).tolist() == [False]
    course = MixedStairCourse(1)
    course.start(torch.tensor([0]), torch.tensor([YAW]), torch.tensor([1]),
                 torch.tensor([2]), torch.tensor([.6]))
    for _ in range(9):
        assert not bool(course.observe_platform_conservative(
            torch.tensor([0]), wheel_centers, hull, *args, torch.tensor([False])))
    assert bool(course.observe_platform_conservative(
        torch.tensor([0]), wheel_centers, hull, *args, torch.tensor([False])))


def test_full_platform_and_command_relative_path_required_for_mixed_promotion():
    # A 4 m radius or single front-wheel landing can no longer promote.
    kwargs = dict(eligible=True, physical_failure=False, lateral_error_m=0.05,
                  yaw_error_rad=0.02, vx_mae_mps=0.1, vy_mae_mps=0.03, wz_mae_rad_s=0.02)
    assert curriculum_decision(completed_platform=False, **kwargs) == (False, True)
    assert curriculum_decision(completed_platform=True, **kwargs) == (True, False)
    assert curriculum_decision(completed_platform=True, **(kwargs | {"lateral_error_m": 0.8})) == (False, True)
    # An intentional lateral/yaw command is scored relative to its integrated path.
    desired = torch.tensor([[3.3, 0.08, 0.05]])
    actual = torch.tensor([[3.3, 0.10, 0.06]])
    assert signed_path_error(actual, desired)[0].tolist() == pytest.approx([0., .02, .01], abs=1e-6)


def test_local_pose_cardinal_rotation_respects_reset_heading():
    origin = torch.tensor([[10., 20.]])
    world = torch.tensor([[10., 23.]])
    yaw90 = torch.tensor([[2**-.5, 0., 0., 2**-.5]])
    local = actual_pose_in_stair_frame(world, yaw90, origin, torch.tensor([1]))
    assert local[0].tolist() == pytest.approx([3., 0., 0.], abs=1e-6)
    course = MixedStairCourse(1)
    course.start(torch.tensor([0]), torch.tensor([YAW]), torch.tensor([1]),
                 torch.tensor([2]), torch.tensor([.6]), torch.tensor([[.08, -.06, .02]]))
    torch.testing.assert_close(course.desired_pose[0], torch.tensor([.08, -.06, .02]))


def test_category_sign_statistics_and_checkpoint_roundtrip_preserve_course():
    course = MixedStairCourse(3)
    ids = torch.tensor([0, 1, 2])
    course.start(ids, torch.tensor([STRAIGHT, YAW, LATERAL]), torch.tensor([0, -1, 1]),
                 torch.tensor([1, 2, 2]), torch.tensor([.5, .6, .7]))
    course.elapsed_s[1:] = 2.2
    issued = scheduled_command(course.group, course.sign, course.speed, course.elapsed_s)
    assert issued[1, 2] < 0 and issued[2, 1] > 0
    course.advance(ids, .02)
    course.observe_tracking(ids, issued, issued)
    points = torch.tensor([[[3., -.2], [3., .2], [3.4, -.2], [3.4, .2]]])
    for _ in range(10):
        course.observe_platform(torch.tensor([1]), points, torch.full((1, 4), 10.),
                                torch.tensor([2.8]), torch.tensor([4.5]), torch.tensor([1.]),
                                torch.tensor([False]))
    course.record(torch.tensor([1]), torch.tensor([1]), torch.tensor([0]), torch.tensor([9]),
                  torch.tensor([True]), torch.tensor([True]), torch.tensor([True]), torch.tensor([False]),
                  torch.tensor([False]), torch.tensor([.02]), torch.zeros((1, 3)))
    assert course.episode_totals[1, 0, 9, YAW, 0, 2, 0] == 1
    host = SimpleNamespace(mixed_stair_course=course)
    payload = checkpoint_payload(host)
    exact = MixedStairCourse(3)
    exact.load_state_dict(payload["mixed_stair_course"])
    for key, value in course.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, exact.state_dict()[key], rtol=0, atol=0)
    reloaded = SimpleNamespace(mixed_stair_course=MixedStairCourse(3))
    assert restore_checkpoint(payload, reloaded)["new_episode_totals"] == "restored exactly"
    torch.testing.assert_close(course.episode_totals, reloaded.mixed_stair_course.episode_totals, rtol=0, atol=0)
    assert not reloaded.mixed_stair_course.active.any()
    assert restore_checkpoint({"stair_command_profile": "straight_speed_v1"}, reloaded)[
        "active_episodes"] == "reset on resume"
    with pytest.raises(ValueError, match="profile"):
        restore_checkpoint({"mixed_stair_profile": "wrong"}, reloaded)


def test_runtime_mixin_reasserts_yaw_and_vy_after_straight_parent_overwrites_command():
    class FrozenStraightParent:
        def _update_command(self):
            self.commands[:] = 0
            self.commands[:, 0] = .6

    class Adapter(MixedStairCommandMixin, FrozenStraightParent):
        def mixed_terminal_mask(self):
            return torch.tensor([True, True, False])

    adapter = Adapter.__new__(Adapter)
    adapter.num_envs = 3
    adapter.device = "cpu"
    adapter.commands = torch.zeros((3, 3))
    adapter.commands_xy_accumulation = torch.zeros((3, 2))
    adapter.cfg = SimpleNamespace(resampling_time=5.)
    adapter.mixed_stair_course = MixedStairCourse(3)
    adapter.mixed_stair_course.start(torch.tensor([0, 1]), torch.tensor([YAW, LATERAL]),
                                     torch.tensor([-1, 1]), torch.tensor([2, 2]), torch.tensor([.6, .6]))
    adapter.mixed_stair_course.elapsed_s[:2] = 2.2
    adapter._update_command()
    assert adapter.commands[0, 2] < 0
    assert adapter.commands[1, 1] > 0
    assert adapter.commands[2].tolist() == pytest.approx([.6, 0., 0.])
    adapter.mixed_stair_course.elapsed_s[:2] = 3.2
    adapter._update_command()
    assert adapter.commands[0, 2] > 0
    assert adapter.commands[1, 1] < 0
    snapshot = checkpoint_payload(adapter)
    resumed = MixedStairCourse(3)
    adapter.mixed_stair_course = resumed
    restore_checkpoint(snapshot, adapter)
    adapter._update_command()
    # The simulator restarted; no mid-flight pulse may survive its new reset.
    assert adapter.commands[:, 1:].abs().sum() == 0


def test_runtime_conservative_provider_records_support_mode_and_fails_closed():
    class Parent:
        def _update_metrics(self):
            pass

    class Adapter(MixedStairCommandMixin, Parent):
        def mixed_terminal_mask(self):
            return torch.tensor([True])

    adapter = Adapter.__new__(Adapter)
    adapter.num_envs, adapter.device = 1, "cpu"
    adapter._env = SimpleNamespace(
        episode_length_buf=torch.tensor([1]), step_dt=.02,
        stair_axis=torch.tensor([0]),
        scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False])))
    adapter.robot = SimpleNamespace(data=SimpleNamespace(
        root_lin_vel_b=torch.zeros((1, 3)), root_ang_vel_b=torch.zeros((1, 3))))
    adapter.commands = torch.tensor([[.6, 0., 0.]])
    adapter.mixed_stair_course = MixedStairCourse(1)
    adapter.mixed_stair_course.start(torch.tensor([0]), torch.tensor([YAW]),
                                     torch.tensor([1]), torch.tensor([2]), torch.tensor([.6]))
    adapter.mixed_platform_start = torch.tensor([2.8])
    adapter.mixed_platform_end = torch.tensor([4.5])
    adapter.mixed_platform_lane = torch.tensor([1.0])
    adapter._mixed_support_contract = None
    data = {
        "kind": "conservative_center_hull",
        "validated_collision_geometry": True,
        "collision_geometry_id": "test_collision_v1",
        "wheel_radius_m": .12,
        "wheel_centers_world_xyz": torch.tensor([[
            [3.15, -.30, .62], [3.15, .30, .62],
            [3.45, -.30, .62], [3.45, .30, .62]]]),
        "platform_world_z_m": torch.tensor([.5]),
        "body_collision_hull_world_xy": torch.tensor([[
            [3.05, -.40], [3.05, .40], [3.60, -.40], [3.60, .40]]]),
        "terrain_forces_n": torch.full((1, 4), 20.),
    }
    adapter._mixed_stair_contact_provider = lambda env, ids: data
    for _ in range(10):
        adapter._update_metrics()
    assert adapter.mixed_stair_course.platform_complete.tolist() == [True]
    assert checkpoint_payload(adapter)["mixed_stair_support_contract"]["kind"] == "conservative_center_hull"
    data["collision_geometry_id"] = "changed_without_migration"
    with pytest.raises(ValueError, match="geometry changed"):
        adapter._update_metrics()
    data["collision_geometry_id"] = "test_collision_v1"
    data["validated_collision_geometry"] = False
    with pytest.raises(ValueError, match="validated collision geometry"):
        adapter._update_metrics()
