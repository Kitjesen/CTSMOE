import torch

from composite_course_state import CompositeCourseState
from composite_route import COMPACT_9X9
from route_acceptance import AcceptanceSpec


def _frame(x):
    wheels = torch.tensor([[[x - .08, 4.2], [x + .08, 4.2],
                            [x - .08, 4.8], [x + .08, 4.8]]], dtype=torch.float32)
    body = torch.tensor([[[x - .08, 4.1], [x + .08, 4.1],
                          [x - .08, 4.9], [x + .08, 4.9]]], dtype=torch.float32)
    return dict(wheel_xy=wheels, body_hull_xy=body,
                wheel_support=torch.ones((1, 4), dtype=torch.bool),
                translation_speed_mps=torch.zeros(1), yaw_rate_rps=torch.zeros(1),
                body_tilt_rad=torch.zeros(1),
                physical_failure=torch.zeros(1, dtype=torch.bool), dt_s=.05)


def test_course_success_needs_real_sections_and_dwell_then_exact_state_restore():
    course = CompositeCourseState(1, acceptance=AcceptanceSpec(dwell_required_s=.1))
    course.set_active(torch.tensor([True]))
    course.update(**_frame(3.0))
    course.update(**_frame(5.1))
    assert course.seen_slope.item() and course.seen_stair.item()
    goal = COMPACT_9X9.final_start_x_m + .2
    assert not course.update(**_frame(goal)).item()
    saved = course.state_dict()
    exact = CompositeCourseState(1, acceptance=AcceptanceSpec(dwell_required_s=.1))
    exact.load_state_dict(saved, environment_state_restored=True)
    assert exact.dwell_s.item() == course.dwell_s.item()
    assert exact.update(**_frame(goal)).item()
    course.record_episodes(torch.tensor([0]), torch.tensor([1]), torch.tensor([False]))
    assert course.totals[1].tolist() == [1, 1, 1, 1, 0, 0]


def test_fresh_physics_resume_preserves_aggregates_but_restarts_episodic_flags():
    source = CompositeCourseState(1)
    source.set_active(torch.tensor([True]))
    source.update(**_frame(3.0))
    source.totals[0, 0] = 17
    saved = source.state_dict()
    resumed = CompositeCourseState(1)
    resumed.set_active(torch.tensor([True]))
    resumed.load_state_dict(saved, environment_state_restored=False)
    assert resumed.totals[0, 0] == 17
    assert not resumed.seen_slope.item()
    assert resumed.dwell_s.item() == 0
    assert resumed.active.item()
