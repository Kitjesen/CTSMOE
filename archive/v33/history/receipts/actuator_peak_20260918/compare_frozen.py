"""Summarize the matched frozen-policy torque-cap ablation, without learning."""
import json
from pathlib import Path


here = Path(__file__).resolve().parent
old = json.loads((here.parent / 'actuator_guard_20260918/heading19001_guard_complete.json').read_text())
new = json.loads((here / 'heading19001_peak.json').read_text())
assert old['checkpoint_sha256'] == new['checkpoint_sha256']
for key in ('completed_updates', 'conditions', 'scope', 'diagnostic_mode', 'heading_feedback',
            'joint_names', 'hard_limits_rad', 'soft_limits_rad', 'velocity_limits_rad_s'):
    assert old[key] == new[key], key
assert old['actuator_profile'] == 'rs0402_guard_v1'
assert new['actuator_profile'] == 'rs0402_peak_v2'
assert len(old['cases']) == len(new['cases']) == 162
for before, after in zip(old['cases'], new['cases']):
    for key in ('learner', 'terrain', 'level', 'speed', 'repeat', 'column'):
        assert before[key] == after[key]
summary = {'checkpoint_sha256': new['checkpoint_sha256'], 'completed_updates': new['completed_updates'],
           'scope': new['scope'], 'conditions': new['conditions'], 'heading_feedback': new['heading_feedback'],
           'student': {}}
for terrain in ('wave', 'stairs_up', 'stairs_down'):
    summary['student'][terrain] = {}
    for label, data in (('v1_35_6', old), ('v2_120_17', new)):
        cases = [c for c in data['cases'] if c['learner'] == 'student' and c['terrain'] == terrain]
        item = {'cases': len(cases), 'exits': int(sum(c['forward_exit'] for c in cases)),
                'physical_failures': int(sum(c['physical_failure'] for c in cases)),
                'by_level': {}}
        for level in (3, 6, 9):
            rows = [c for c in cases if c['level'] == level]
            item['by_level'][str(level)] = {
                'exits': int(sum(c['forward_exit'] for c in rows)),
                'physical_failures': int(sum(c['physical_failure'] for c in rows)),
                'timeouts': sum(not c['forward_exit'] and not c['physical_failure'] for c in rows)}
        for kind in ('hip', 'thigh', 'calf', 'foot'):
            indices = [i for i, name in enumerate(data['joint_names']) if name.endswith(kind + '_joint')]
            item[kind] = {key: max(c['joints'][key][i] for c in cases for i in indices)
                          for key in ('max_applied_torque_nm', 'max_speed_rad_s', 'max_hard_excess_rad')}
        summary['student'][terrain][label] = item
summary['v2_student_up_level9'] = [
    {key: c[key] for key in ('speed', 'repeat', 'forward_exit', 'physical_failure', 'duration_s', 'final_forward_m')}
    for c in new['cases'] if c['learner'] == 'student' and c['terrain'] == 'stairs_up' and c['level'] == 9]
(here / 'frozen_comparison.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
