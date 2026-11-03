"""Check that startup conclusions use finite, correctly indexed first episodes."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
a=parser.parse_args()
m=json.loads((a.directory/'manifest.json').read_text())
with np.load(a.directory/'traces.npz') as f:
    assert len(set(m['joint_names']))==16
    assert len(m['cases'])==f['joint_pos'].shape[1]
    for c in m['cases']:
        n=c['end_step']+1
        i=c['id']
        assert n<=f['joint_pos'].shape[0]
        for key in ('joint_pos','joint_vel','target','torque','velocity','root_clearance','raw_action'):
            assert np.isfinite(f[key][:n,i]).all(), (i,key)
        assert f['active'][:n,i].all(), i
        start=100 if c['mode']=='settle' else 0
        if start:
            np.testing.assert_allclose(f['command'][:min(n,start),i],0.,atol=1e-6)
        if n>start:
            np.testing.assert_allclose(f['command'][start:n,i],
                np.broadcast_to(c['velocity'],(n-start,3)),atol=1e-6)
(a.directory/'validation.json').write_text(json.dumps({'valid':True,'cases':len(m['cases'])}))
print('TRACE_VALID',len(m['cases']))
