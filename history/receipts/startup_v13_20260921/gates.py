"""Matched-case continuation criteria; never use training reward as acceptance."""
import math


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


def flat25_gate(analysis, baseline_label, candidate_label, final=False):
    """Reject delayed failure even when the first second looks better.

    Thresholds are registered before V12 training: retain baseline completions,
    improve forward startup, and keep the full forward trace below 0.35 rad RMS.
    Final acceptance additionally requires all 20 directions/start modes to pass.
    """
    baseline, candidate = [[c for c in analysis['cases'] if c['label'] == label]
                           for label in (baseline_label, candidate_label)]
    if len(baseline) != 20 or len(candidate) != 20:
        raise ValueError('Long flat protocol requires 20 cases per policy')
    directions = ('forward', 'reverse', 'left', 'right', 'yaw_left', 'yaw_right',
                  'diagonal', 'diagonal_back', 'forward_turn', 'side_turn')
    expected = {(mode, direction) for mode in ('direct', 'settle') for direction in directions}
    for rows in (baseline, candidate):
        if {(c['mode'], c['direction']) for c in rows} != expected:
            raise ValueError('Long flat protocol has duplicate or missing direction/start modes')
    reasons = []
    for old, new in zip(baseline, candidate):
        keys = ('terrain', 'mode', 'direction', 'level', 'velocity')
        if any(old[k] != new[k] for k in keys) or new['terrain'] != 'flat':
            raise ValueError('Long flat case ordering or commands changed')
        name = f"{new['direction']}/{new['mode']}"
        values = ('first_peak_rms', 'first_peak_joint_rad', 'max_rms', 'last_xy_error', 'last_yaw_error')
        if any(old[k] is not None and not math.isfinite(old[k]) for k in values):
            raise ValueError(f'{name}: nonfinite baseline evidence')
        if any(new[k] is None or not math.isfinite(new[k]) for k in values):
            reasons.append(f'{name}: incomplete or nonfinite trajectory')
            continue
        if old['completed_25s'] and not new['completed_25s']:
            reasons.append(f'{name}: lost a full 25 second completion')
        if new['physical'] and not old['physical']:
            reasons.append(f'{name}: new physical failure')
        if new['last_xy_error'] > max(.15, old['last_xy_error'] or 0.)+.02:
            reasons.append(f'{name}: XY tracking regressed')
        if new['last_yaw_error'] > max(.2, old['last_yaw_error'] or 0.)+.02:
            reasons.append(f'{name}: yaw tracking regressed')
        if new['max_rms'] > max(.35, old['max_rms'])+.02:
            reasons.append(f'{name}: full-episode posture regressed')
        if new['direction'] == 'forward':
            threshold = .25 if final or old['first_peak_rms'] <= .25 else .8*old['first_peak_rms']
            if new['first_peak_rms'] >= threshold:
                reasons.append(f'{name}: startup improvement insufficient')
            if new['max_rms'] >= .35 or not new['completed_25s']:
                reasons.append(f'{name}: delayed posture excursion or incomplete 25 seconds')
        if final and not new['pass']:
            reasons.append(f'{name}: final motion acceptance failed')
    return dict(accepted=not reasons, reasons=reasons, baseline=baseline_label, candidate=candidate_label,
                final=final, protocol='flat25_80m', fixed_cases_not_random_success_rate=True,
                baseline_completions=sum(c['completed_25s'] for c in baseline),
                candidate_completions=sum(c['completed_25s'] for c in candidate))
