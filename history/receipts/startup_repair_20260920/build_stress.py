"""Derive an explicitly perturbed startup validation from the nominal probe."""
from pathlib import Path

p=Path(__file__).parent
s=(p/'audit.py').read_text()
changes={
    "selection=json.loads(a.selection.read_text())":"selection=json.loads(a.selection.read_text())[:2]",
    "teacher=item.get('teacher',False)) for index,item in enumerate(selection)":
    "teacher=item.get('teacher',False),replicate=replicate) for index,item in enumerate(selection)",
    "for mode in ('direct','settle') for name,velocity in commands]":
    "for replicate in range(3) for mode in ('direct','settle') for name,velocity in commands]",
    "group.enable_corruption=False":"group.enable_corruption=True",
    "position_range=(1.,1.),velocity_range=(0.,0.)":"position_range=(.98,1.02),velocity_range=(0.,0.)",
    "actuator.min_delay=actuator.max_delay=2":"actuator.min_delay,actuator.max_delay=0,4",
    "Matched nominal flat L0, default pose and zero initial velocities; fixed10ms delay, no noise/push;":
    "Perturbed flat L0: joint pose scale0.98..1.02, configured observation noise enabled, random actuator delay0..20ms, no pushes; three samples per condition with independent noise;",
}
for old,new in changes.items():
    assert s.count(old)==1, old
    s=s.replace(old,new)
(p/'stress.py').write_text(s)
