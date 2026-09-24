"""Extract the recorded 45-degree slope failure without using the GPU."""
import json
from pathlib import Path
import numpy as np

SOURCE = Path('/root/autodl-tmp/cts-v15-evaluation-audit-20260922/terrain83')
OUTPUT = Path(__file__).resolve().parent / 'overspeed_event.json'


def main():
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    rows = json.loads((SOURCE / 'analysis_v14.json').read_text())['cases']
    case = next(row for row in rows if row['label'] == 'V15_32131'
                and row['terrain'] == 'slope_up' and row['level'] == 9
                and row['velocity'] == [.3, 0., 0.])
    case_id, n = case['id'], case['end_step'] + 1
    with np.load(SOURCE / 'traces.npz') as archive:
        keys = ['joint_vel', 'speed_peak', 'torque', 'torque_peak', 'command',
                'velocity', 'angular_velocity', 'wheel_clearance', 'wheel_ground_force',
                'body_pos', 'body_quat', 'raw_action', 'base_contact']
        data = {key: archive[key][:n, case_id].copy() for key in keys}
    names = manifest['joint_names']
    wheels = [names.index(f'{side}_foot_joint') for side in ('fr', 'fl', 'rr', 'rl')]
    base = manifest['body_names'].index('base_link')
    t, j = np.unravel_index(data['speed_peak'].argmax(), data['speed_peak'].shape)
    records = []
    for step in range(max(0, t - 25), min(n, t + 1)):
        q = data['body_quat'][step, base]
        w, x, y, z = q
        rpy = [np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
               np.arcsin(np.clip(2*(w*y-z*x), -1., 1.)),
               np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))]
        records.append(dict(time_s=(step+1)*manifest['dt'],
            joint_speed=data['joint_vel'][step, j].item(),
            joint_speed_substep_peak=data['speed_peak'][step, j].item(),
            joint_torque=data['torque'][step, j].item(),
            joint_torque_substep_peak=data['torque_peak'][step, j].item(),
            wheel_speeds=data['joint_vel'][step, wheels].tolist(),
            wheel_forces=data['wheel_ground_force'][step].tolist(),
            wheel_clearance=data['wheel_clearance'][step].tolist(),
            root_pos=data['body_pos'][step, base].tolist(),
            body_velocity=data['velocity'][step].tolist(),
            angular_velocity=data['angular_velocity'][step].tolist(),
            rpy_deg=np.rad2deg(rpy).tolist(),
            wheel_unclipped_request=(data['raw_action'][step, -4:]*5).tolist(),
            base_contact=data['base_contact'][step].item()))
    result = dict(case=case, joint=names[j], event_time_s=(t+1)*manifest['dt'],
                  speed_peak_rad_s=float(data['speed_peak'][t, j]),
                  limits=manifest['actuator_limits'],
                  contact_order=['fr', 'fl', 'rr', 'rl'], frames=records,
                  observation_limit='20 ms state/contact sampling; speed peaks include 5 ms physics substeps. Exact within-step contact causality needs a substep replay.')
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != 'frames'}))
    print(json.dumps(records[-8:]))


if __name__ == '__main__':
    main()
