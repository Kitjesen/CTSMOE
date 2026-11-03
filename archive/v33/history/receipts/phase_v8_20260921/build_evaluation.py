"""Add V8 terrain/reward context to the existing fixed-case evaluation."""
from pathlib import Path

b=Path(__file__).parent
source=(b.parent/'startup_repair_20260920/all_scene_phase_audit.py').read_text(encoding='utf-8')
old="    configure_stance_strength(cfg,latest['stance_strength_profile'])"
assert source.count(old)==1
source=source.replace(old,old+"""
    from startup_repair import configure as configure_startup
    from phase_transition import measure as phase_measure
    assert latest['startup_repair_profile']=='omni_start_v8'
    configure_startup(cfg,'omni_start_v8')
    cfg.terminations.startup_window=None
""")
source=source.replace("'platform_stop')}","'platform_stop','phase_weight','phase_ready','phase_stable_s')}")
source=source.replace('            values=dict(ground_relief=relief,ground_scan_valid=valid,',
    "            phase=phase_measure(raw)\n            values=dict(phase_weight=phase['weight'],phase_ready=phase['ready'],phase_stable_s=phase['stable_s'],ground_relief=relief,ground_scan_valid=valid,")
source=source.replace("protocol='Nine terrains,", "protocol='V8 landing geometry (shorter stair flight, outer plateau); phase telemetry enabled. Nine terrains,")
source=source.replace('for level in (9,15,21) for speed', 'for level in (0,2,4,6,9,15,21) for speed')
source=source.replace('assert len(cases)==71*len(selection)', 'assert len(cases)==83*len(selection)')
source=source.replace('71 cases per policy:', '83 cases per policy:')
source=source.replace('pit0.6/0.9/1.2m', 'pit0.15/0.25/0.35/0.45/0.6/0.9/1.2m')
(b/'evaluate_landing.py').write_text(source,encoding='utf-8',newline='\n')
