"""Add explicitly measured landing and target-step evidence without replacing reports."""
import argparse
import json
from pathlib import Path
import numpy as np

def enrich(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    analysis=json.loads((directory/'analysis.json').read_text())
    base=manifest['body_names'].index('base_link')
    legs=np.array(manifest['request_joint_ids'])
    with np.load(directory/'traces.npz') as traces:
        for row in analysis['cases']:
            i,n=row['id'],row['end_step']+1
            target=traces['target'][:n,i][:,legs]
            initial=traces['initial_q'][i,legs]
            delta=np.diff(target,axis=0,prepend=initial[None,:])
            row['target_step_peak_rad']=float(np.abs(delta).max())
            if row['terrain'].startswith('stairs'):
                quat=traces['body_quat'][:n,i,base]
                velocity=traces['velocity'][:n,i]
                vector=quat[:,1:]
                world=velocity+2*(quat[:,:1]*np.cross(vector,velocity)+np.cross(vector,np.cross(vector,velocity)))
                vz=world[-100:,2]
                row.update(end_vz_abs_peak=float(np.abs(vz).max()),end_vz_min=float(vz.min()),end_vz_max=float(vz.max()))
    analysis['evidence_revision']='v14_landing_abs_and_all_leg_target_steps'
    with (directory/'analysis_v14.json').open('x') as f: json.dump(analysis,f,indent=2)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);enrich(p.parse_args().directory)

