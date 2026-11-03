"""Static A/B geometry checks; these do not establish loaded stance stability."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from thunder_lab3.assets.contract import (A_POSE,A_POSE_CONTACT_HEIGHT,A_POSE_SPAWN_HEIGHT,
    B_POSE,B_POSE_CONTACT_HEIGHT,B_POSE_SPAWN_HEIGHT,POSE,validate_urdf)
from tools.audits.geometry_audit import URDF_SHA256, rotate, transform, vec


URDF = Path(__file__).resolve().parents[1] / "source/thunder_lab3/assets/thunder/urdf/thunder_v4.urdf"


class PoseCandidateTests(unittest.TestCase):
    def test_candidate_keeps_asset_hips_and_all_soft_limits(self):
        report = validate_urdf(URDF, pose=B_POSE)
        self.assertEqual(report["sha256"], URDF_SHA256)
        self.assertAlmostEqual(report["total_urdf_mass_kg"], 45.8086)
        self.assertEqual(set(B_POSE), set(POSE))
        for name, q in B_POSE.items():
            if name.endswith(("_hip_joint", "_foot_joint")):
                self.assertEqual(q, POSE[name])
            if name in report["joint_position_limits"]:
                low, high = report["joint_position_limits"][name]
                center, half_range = (low + high) / 2, .9 * (high - low) / 2
                self.assertGreaterEqual(q, center - half_range, name)
                self.assertLessEqual(q, center + half_range, name)
        self.assertAlmostEqual(B_POSE_SPAWN_HEIGHT - B_POSE_CONTACT_HEIGHT, .02)

    @staticmethod
    def geometry(pose):
        root = ET.parse(URDF).getroot()
        bottoms, inclinations, knees = [], [], []
        for leg in ("fr", "fl", "rr", "rl"):
            current, anchors = np.eye(4), {}
            for part in ("hip", "thigh", "calf", "foot"):
                joint = root.find(f"joint[@name='{leg}_{part}_joint']")
                current = current @ transform(joint.find("origin"))
                anchors[part] = current[:3, 3].copy()
                motion = np.eye(4)
                motion[:3, :3] = rotate(vec(joint.find("axis")), pose[f"{leg}_{part}_joint"])
                current = current @ motion
            collision = root.find(f"link[@name='{leg}_foot_Link']/collision")
            frame = current @ transform(collision.find("origin"))
            cylinder = collision.find("geometry/cylinder")
            axis_z = frame[2, 2]
            extent = (float(cylinder.get("radius")) * np.sqrt(max(0., 1. - axis_z**2))
                      + float(cylinder.get("length")) * abs(axis_z) / 2)
            bottoms.append(frame[2, 3] - extent)
            thigh = anchors["calf"] - anchors["thigh"]
            inclinations.append(np.degrees(np.arctan2(thigh[0], -thigh[2])))
            upper=thigh[[0,2]];lower=(anchors['foot']-anchors['calf'])[[0,2]]
            knees.append(np.degrees(np.arccos(np.clip(np.dot(-upper,lower)/np.linalg.norm(upper)/np.linalg.norm(lower),-1,1))))
        return np.array(bottoms),np.array(inclinations),np.array(knees)

    def test_strict_a_matches_go2_angles_without_claiming_level_wheels(self):
        report=validate_urdf(URDF,pose=A_POSE)
        self.assertEqual(report['sha256'],URDF_SHA256)
        bottoms,inclinations,knees=self.geometry(A_POSE)
        np.testing.assert_allclose(inclinations,[-45.98003016604069]*2+[-57.42610812106981]*2,atol=1e-9,rtol=0.)
        np.testing.assert_allclose(knees,[93.77147338813516]*2+[93.80510571610398]*2,atol=1e-9,rtol=0.)
        self.assertAlmostEqual(-bottoms.min(),A_POSE_CONTACT_HEIGHT)
        self.assertAlmostEqual(np.ptp(bottoms),.01168787076265726)
        self.assertAlmostEqual(A_POSE_SPAWN_HEIGHT-A_POSE_CONTACT_HEIGHT,.02)
        self.assertGreater(bottoms[2],bottoms[0]+.011)

    def test_wheel_collision_bottoms_are_level_at_candidate_contact_height(self):
        bottoms,inclinations,_=self.geometry(B_POSE)
        np.testing.assert_allclose(bottoms, -B_POSE_CONTACT_HEIGHT, atol=1e-10, rtol=0.)
        np.testing.assert_allclose(inclinations, [-45.980, -45.980, -57.426, -57.426], atol=.002, rtol=0.)
        # The 0.4 mm FL calf-origin offset requires unequal raw front angles.
        self.assertGreater(abs(B_POSE["fr_thigh_joint"] + B_POSE["fl_thigh_joint"]), .001)


if __name__ == "__main__":
    unittest.main()
