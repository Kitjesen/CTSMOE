"""Matched-case continuation criteria; never use training reward as acceptance."""


def policy_cases(analysis, label):
    result = [c for c in analysis['cases'] if c['label'] == label]
    if len(result) != 83:
        raise ValueError(f'Expected 83 complete cases for {label}, got {len(result)}')
    return result


def skill_regressions(analysis, baseline_label, candidate_label):
    baseline, candidate = [policy_cases(analysis, x) for x in (baseline_label, candidate_label)]
    reasons = []
    for terrain in sorted({c['terrain'] for c in baseline}):
        if terrain == 'flat':
            continue
        old = [c for c in baseline if c['terrain'] == terrain]
        new = [c for c in candidate if c['terrain'] == terrain]
        if len(old) != len(new):
            raise ValueError('Case distribution changed')
        if sum(c['pass'] for c in new) < sum(c['pass'] for c in old):
            reasons.append(f'{terrain}: fewer accepted cases')
        if sum(c['physical'] for c in new) > sum(c['physical'] for c in old):
            reasons.append(f'{terrain}: more physical failures')
        if terrain.startswith('stairs') and sum(c['straight_4m'] for c in new) < sum(c['straight_4m'] for c in old):
            reasons.append(f'{terrain}: fewer straight traversals')
        if terrain == 'pit':
            old_keys = {(c['level'], tuple(c['velocity'])) for c in old if c['pass']}
            new_keys = {(c['level'], tuple(c['velocity'])) for c in new if c['pass']}
            if old_keys-new_keys:
                reasons.append('pit: a previously successful height/speed case regressed')
    return reasons


def continuation_gate(analysis, baseline_label, candidate_label, final=False):
    baseline, candidate = [policy_cases(analysis, x) for x in (baseline_label, candidate_label)]
    reasons = skill_regressions(analysis, baseline_label, candidate_label)
    old_flat = [c for c in baseline if c['terrain']=='flat']
    new_flat = [c for c in candidate if c['terrain']=='flat']
    if sum(c['pass'] for c in new_flat) < sum(c['pass'] for c in old_flat):
        reasons.append('flat: fewer accepted cases')
    if sum(c['physical'] for c in new_flat) > sum(c['physical'] for c in old_flat):
        reasons.append('flat: more physical failures')
    def forward(rows):
        return next(c for c in rows if c['terrain']=='flat' and c['mode']=='direct' and c['direction']=='forward')
    b, n = forward(baseline), forward(candidate)
    limit = .25 if final or b['first_peak_rms'] <= .25 else .8*b['first_peak_rms']
    if n['first_peak_rms'] >= limit:
        reasons.append('forward startup RMS did not meet improvement criterion')
    if final and n['first_peak_joint_rad'] >= .6:
        reasons.append('forward single-joint excursion remains excessive')
    for old, new in zip(baseline, candidate):
        if (old['terrain'], old['mode'], old['direction'], old['level'], old['velocity']) != (
                new['terrain'], new['mode'], new['direction'], new['level'], new['velocity']):
            raise ValueError('Case ordering or commands changed')
        if old['terrain'] == 'flat':
            if new['last_xy_error'] > max(.15, old['last_xy_error'])+.02:
                reasons.append(f"flat/{new['direction']}/{new['mode']}: XY tracking regressed")
            if new['last_yaw_error'] > max(.2, old['last_yaw_error'])+.02:
                reasons.append(f"flat/{new['direction']}/{new['mode']}: yaw tracking regressed")
            if final and not new['pass']:
                reasons.append(f"flat/{new['direction']}/{new['mode']}: not accepted")
    return dict(accepted=not reasons, reasons=reasons, baseline=baseline_label, candidate=candidate_label,
                forward_rms_before=b['first_peak_rms'], forward_rms_after=n['first_peak_rms'],
                threshold=limit, final=final, fixed_cases_not_random_success_rate=True)
