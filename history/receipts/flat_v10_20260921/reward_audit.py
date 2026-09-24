"""Recompute candidate cost on saved trajectories; this is not a policy test."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

stage = Path(__file__).resolve().parent
sys.path.insert(0, str(stage.parent.parent))
from flat_outlier import components  # noqa: E402

source = stage.parent/'flat_v9_20260921/interim_44001'
m = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
d = np.load(source/'forward_traces.npz')
legs = [i for i, name in enumerate(m['joint_names']) if not name.endswith('_foot_joint')]
scales = torch.tensor([next(v for k, v in dict(hip=.25, thigh=.45, calf=.55).items()
                          if m['joint_names'][i].endswith('_'+k+'_joint')) for i in legs])
q0 = torch.tensor(m['default_q'])[legs]
rows = []
for column, case_id in enumerate(d['ids']):
    case = next(c for c in m['cases'] if c['id'] == case_id)
    n = min(50, case['end_step']+1)
    q = torch.tensor(d['joint_pos'][:n, column][:, legs])
    mean, tail = components(q, q0, scales)
    peak = int((q-q0).square().mean(-1).argmax())
    rows.append(dict(label=m['history'][case['policy']]['label'],
                     first_second_old_cost=float((2*mean).mean()),
                     first_second_new_cost=float((2*mean+tail).mean()),
                     at_pose_peak_old_cost=float(2*mean[peak]),
                     at_pose_peak_new_cost=float(2*mean[peak]+tail[peak])))
receipt = dict(kind='counterfactual_reward_only_not_behavior_improvement',
               excludes_environment_dt_factor=True, cases=rows)
(stage/'reward_counterfactual.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
