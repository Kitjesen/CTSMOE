from types import SimpleNamespace

import torch

from mixed_stair_logic import LATERAL, MixedStairCourse
from mixed_stair_runtime import PROFILE, restore_checkpoint


def test_restart_preserves_group_statistics_but_discards_old_physical_episode():
    source = MixedStairCourse(2)
    source.start(torch.tensor([0]), torch.tensor([LATERAL]), torch.tensor([1]),
                 torch.tensor([2]), torch.tensor([0.7]))
    source.elapsed_s[0] = 2.4
    source.support_streak[0] = 3
    source.episode_totals[1, 0, 4, LATERAL, 2, 2, 0] = 7
    saved = {"mixed_stair_profile": PROFILE, "mixed_stair_course": source.state_dict(),
             "mixed_stair_support_contract": {"kind": "conservative_center_hull",
                                              "collision_geometry_id": "test"}}
    target = SimpleNamespace(mixed_stair_course=MixedStairCourse(2), _mixed_support_contract=None)
    receipt = restore_checkpoint(saved, target)
    assert receipt["active_episodes"] == "reset on resume"
    assert target.mixed_stair_course.episode_totals[1, 0, 4, LATERAL, 2, 2, 0] == 7
    assert not target.mixed_stair_course.active.any()
    assert target.mixed_stair_course.elapsed_s.sum() == 0
    assert target.mixed_stair_course.support_streak.sum() == 0
    assert target._mixed_support_contract == saved["mixed_stair_support_contract"]
