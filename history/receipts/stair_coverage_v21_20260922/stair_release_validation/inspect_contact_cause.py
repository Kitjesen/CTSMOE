"""Summarize exact contact replay at the three already identified peaks."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "contact_review"
results = []
for condition, case_ids in (("nominal", [28, 64]), ("perturbed", [33])):
    folder = ROOT / condition
    meta = json.loads((folder / "CONTACT_CAPTURE.json").read_text())
    events = json.loads((folder / "CONTACT_EVENTS.json").read_text())
    with np.load(folder / "CONTACT_CAUSE_5MS.npz") as archive:
        z = dict(archive)
    assert meta["baseline_outcomes_identical"]
    assert not any(meta["baseline_max_abs_difference"].values())
    for case_id in case_ids:
        local_id = meta["case_ids"].index(case_id)
        peak_time = {28: 5.605, 64: 1.660, 33: 2.620}[case_id]
        side = "fl" if case_id == 64 else "fr"
        joint = meta["joint_names"].index(side + "_calf_joint")
        body = meta["body_names"].index(side + "_foot_Link")
        origin = np.asarray(meta["env_origins"][local_id])
        at = int(round(peak_time / .005)) - 1
        rows = []
        # Case33 terminates at the peak. Later samples use inactive-case zero
        # actions and are not part of its original motion episode.
        for index in range(at - 4, at + (1 if case_id == 33 else 5)):
            time = (index + 1) * .005
            contact = [e for e in events if e["case_id"] == case_id and e["wheel"] == side
                       and abs(e["time_s"] - time) < 1e-7]
            assert len(contact) == 1
            contact = contact[0]
            points = np.asarray(contact["point_w"]).reshape(-1, 3)
            normals = np.asarray(contact["normal_w"]).reshape(-1, 3)
            forces = np.asarray(contact["force_N"]).reshape(-1, 1)
            aggregate = (forces * normals).sum(0)
            matrix = np.asarray(contact["matrix_N"])
            body_state = z["body_state"][index, local_id, body]
            top = np.argsort(-forces.ravel())[:3]
            rows.append(dict(time_s=time, q_rad=float(z["q"][index, local_id, joint]),
                target_rad=float(z["target"][index, local_id, joint]),
                dq_rad_s=float(z["dq"][index, local_id, joint]),
                ddq_rad_s2=float(z["ddq"][index, local_id, joint]),
                torque_nm=float(z["torque"][index, local_id, joint]),
                computed_nm=float(z["computed"][index, local_id, joint]),
                wheel_pos_local_m=(body_state[:3] - origin).tolist(),
                wheel_linear_velocity_w=body_state[7:10].tolist(),
                wheel_ground_force_N=matrix.tolist(),
                detailed_contact_count=contact["count"],
                force_reconstruction_error_N=float(abs(aggregate-matrix).max()),
                strongest_contacts=[dict(force_N=float(forces[i, 0]),
                    point_local_m=(points[i]-origin).tolist(), normal=normals[i].tolist(),
                    separation_m=float(np.asarray(contact["separation_m"])[i, 0])) for i in top]))
        result = dict(condition=condition, case_id=case_id, peak_time_s=peak_time,
                      joint=meta["joint_names"][joint], hard_limits=meta["hard_limits"][joint], rows=rows)
        results.append(result)
        print(json.dumps(result, indent=2))
(ROOT / "CONTACT_CAUSE_EVENTS.json").write_text(json.dumps(results, indent=2))
