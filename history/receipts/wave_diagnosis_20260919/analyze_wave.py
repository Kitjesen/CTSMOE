"""Summarize wave ablations and the first second of 200 Hz contact/velocity data."""

import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent
output = {}
for mode in ('baseline', 'rate12', 'no_stop', 'lower_reset', 'history21001'):
    folder = root / mode
    if not (folder / 'results.json').exists():
        continue
    result = json.loads((folder / 'results.json').read_text())
    geometry = json.loads((folder / 'initial_geometry.json').read_text())
    with np.load(folder / 'early_200hz.npz') as source:
        early, fields = source['values'], list(source['fields'])
    with np.load(folder / 'stair_failure_traces.npz') as source:
        trace = {key: source[key] for key in source.files}
    names = result['joint_names']
    leg_ids = [names.index(str(name)) for name in trace['leg_action_joint_names']]
    soft = np.asarray(result['soft_limits_rad'])[leg_ids]
    active = trace['active_before_step'].astype(bool)
    prior = np.clip(trace['previous_leg_target_rad'], soft[:, 0], soft[:, 1])
    request = np.clip(trace['requested_leg_target_rad'], soft[:, 0], soft[:, 1])
    target = trace['joint_pos_target_rad'][:, :, leg_ids]
    limit = (result['ablation']['target_rate'] or 6.0) * .02
    expected = np.clip(prior + np.clip(request-prior, -limit, limit), soft[:, 0], soft[:, 1])
    np.testing.assert_allclose(target[active], expected[active], atol=2e-6, rtol=0)
    rate_limited = np.abs(request-prior) > limit + 1e-5
    speed_columns = [fields.index('speed_' + name) for name in names]
    budgets = np.asarray([30.6 if n.endswith('foot_joint') else 15.3 for n in names])
    details = []
    for i, case in enumerate(result['cases']):
        valid = early[:, i, fields.index('active')].astype(bool)
        samples = early[valid, i]
        wheel_cols = [k for k, f in enumerate(fields) if f.startswith('contact_') and f != 'contact_base']
        def first(mask):
            indices = np.flatnonzero(mask)
            return int(indices[0]) if len(indices) else None
        touch = first(samples[:, wheel_cols].max(-1) > 1)
        fast = first((np.abs(samples[:, speed_columns]) > budgets).any(-1))
        base = first(samples[:, fields.index('contact_base')] > 1)
        details.append(dict(case=i, learner=case['learner'], level=case['level'], speed=case['speed'],
                            repeat=case['repeat'], passed=bool(case['forward_exit']), duration_s=case['duration_s'],
                            reasons=case['termination_reasons'], final_forward_m=case['final_forward_m'],
                            max_roll_pitch_deg=case['max_roll_or_pitch_deg'],
                            wheel_gaps_m=geometry['wheel_bottom_vertical_gap_m'][i],
                            first_wheel_contact_s=(touch+1)*.005 if touch is not None else None,
                            first_speed_violation_s=(fast+1)*.005 if fast is not None else None,
                            first_base_contact_s=(base+1)*.005 if base is not None else None,
                            first_fast_joints={n:float(samples[fast, speed_columns[j]]) for j,n in enumerate(names)
                                               if abs(samples[fast, speed_columns[j]]) > budgets[j]} if fast is not None else {},
                            at_first_fast_root_delta_z_m=float(samples[fast, fields.index('dz')]) if fast is not None else None,
                            at_first_fast_body_vz=float(samples[fast, fields.index('vz')]) if fast is not None else None,
                            peak_early_base_contact_n=float(samples[:, fields.index('contact_base')].max())))
    groups = {}
    for learner in ('teacher', 'student'):
        ids = [i for i,c in enumerate(result['cases']) if c['learner'] == learner]
        rows = [details[i] for i in ids]
        valid = active[:, ids]
        initial = valid & (trace['time_s'][:, None] <= .300001)
        groups[learner] = dict(n=len(rows), passes=sum(c['passed'] for c in rows),
                               by_level={str(level):dict(n=9, passes=sum(c['passed'] for c in rows if c['level']==level),
                                  actuator=sum(c['reasons']['actuator_violation'] for c in rows if c['level']==level),
                                  base_contact=sum(c['reasons']['illegal_contact'] for c in rows if c['level']==level),
                                  timeout=sum(c['reasons']['time_out'] for c in rows if c['level']==level)) for level in (0,3,6,9)},
                               early_failures=sum(not c['passed'] and c['duration_s']<.35 for c in rows),
                               actuator=sum(c['reasons']['actuator_violation'] for c in rows),
                               base_contact=sum(c['reasons']['illegal_contact'] for c in rows),
                               rate_limited_fraction=float(rate_limited[:, ids][valid].mean()),
                               initial_rate_limited_fraction=float(rate_limited[:, ids][initial].mean()))
    output[mode] = dict(checkpoint=result['completed_updates'], ablation=result['ablation'],
                        contract_profile=result['actuator_profile'], actual_projection_verified=True,
                        groups=groups, details=details)
(root / 'analysis.json').write_text(json.dumps(output, indent=2))
print(json.dumps({name:value['groups'] for name,value in output.items()}, indent=2))
