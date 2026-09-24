import copy
import json
from pathlib import Path

from gates import continuation_gate


def matched():
    path = Path(__file__).parent.parent/'flat_v10_20260921/baseline_v9_44711/analysis.json'
    old = [c for c in json.loads(path.read_text(encoding='utf-8'))['cases'] if c['label']=='V9_44711']
    new = copy.deepcopy(old)
    for row in new:
        row['label'] = 'candidate'
        if row['terrain']=='flat' and row['direction']=='forward' and row['mode']=='direct':
            row['first_peak_rms'] = .2
            row['first_peak_joint_rad'] = .5
    return dict(cases=old+new), new


def test_improvement_requires_no_lost_skill():
    analysis, new = matched()
    assert continuation_gate(analysis, 'V9_44711', 'candidate')['accepted']
    pit = next(c for c in new if c['terrain']=='pit' and c['pass'])
    pit['pass'] = False
    result = continuation_gate(analysis, 'V9_44711', 'candidate')
    assert not result['accepted']
    assert any('previously successful' in reason for reason in result['reasons'])


def test_good_forward_alone_does_not_pass_final():
    analysis, _ = matched()
    assert not continuation_gate(analysis, 'V9_44711', 'candidate', final=True)['accepted']


def test_tracking_regression_rejects_candidate():
    analysis, new = matched()
    new[0]['last_xy_error'] = 1.
    assert not continuation_gate(analysis, 'V9_44711', 'candidate')['accepted']
