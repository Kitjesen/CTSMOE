"""Extend the validated read-only probe to measure target slew on all terrains."""
from pathlib import Path


here = Path(__file__).resolve().parent
source = (here.parent / 'stair_stall_followup_20260918/probe_request.py').read_text()
changes = {
    '        capture["requested"] = requested.clone()': '        capture["requested"] = requested.clone()\n        capture["previous"] = previous.clone()',
    '        stair_trace_ids = [i for i, c in enumerate(cases) if c["learner"] == "student"\n                           and c["terrain"] in ("stairs_up", "stairs_down")]': '        stair_trace_ids = [i for i, c in enumerate(cases) if c["learner"] == "student"]',
    '        assert count == 162 and len(stair_trace_ids) == 54 and len(names) == 16': '        assert count == 162 and len(stair_trace_ids) == 81 and len(names) == 16',
    '            "requested_leg_target_rad": (12,),': '            "requested_leg_target_rad": (12,),\n            "previous_leg_target_rad": (12,),',
    '                    "requested_leg_target_rad": capture["requested"],': '                    "requested_leg_target_rad": capture["requested"],\n                    "previous_leg_target_rad": capture["previous"],',
}
for before, after in changes.items():
    assert source.count(before) == 1, before
    source = source.replace(before, after)
(here / 'probe_motion.py').write_text(source)

