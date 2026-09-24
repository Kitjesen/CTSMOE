"""Measure the proposed raw-request cost on the retained climbing policy."""
from pathlib import Path

p=Path(__file__).parent
s=(p/'regression.py').read_text()
changes={
    "from pit_posture import configure as configure_pit_posture":
    "from pit_posture import configure as configure_pit_posture, measure as pit_posture_measure",
    "'climb','stable','active')}":"'climb','stable','active','raw_action','pit_posture_gate')}",
    "climb=pit.climb_seen,stable=pit.success,active=active)":
    "climb=pit.climb_seen,stable=pit.success,active=active,raw_action=actions,\n                        pit_posture_gate=pit_posture_measure(pit)['weight'])",
}
for old,new in changes.items():
    assert s.count(old)==1,old
    s=s.replace(old,new)
(p/'cost_probe.py').write_text(s)
