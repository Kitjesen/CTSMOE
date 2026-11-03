"""Check the shipped Thunder asset without a simulator: python -m unittest tests.test_robot_asset -v."""
import copy
import math
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET

from thunder_lab3.assets.contract import POSE, validate_urdf


URDF = Path(__file__).resolve().parents[1] / "source/thunder_lab3/assets/thunder/urdf/thunder_v4.urdf"


class RobotAssetTests(unittest.TestCase):
    def test_meshes_joint_types_and_reference_pose(self):
        report = validate_urdf(URDF)
        self.assertEqual(report["mesh_count"], 21)
        self.assertEqual(len(report["joint_names"]), 16)
        self.assertEqual(report["root_link"], "base_link")
        self.assertAlmostEqual(report["total_urdf_mass_kg"], 45.8086)
        for name, limits in report["joint_position_limits"].items():
            self.assertLessEqual(limits[0], POSE[name])
            self.assertLessEqual(POSE[name], limits[1])
        self.assertEqual(len(report["joint_position_limits"]), 12)

    def test_asymmetric_calf_limits_are_not_silently_widened(self):
        limits = validate_urdf(URDF)["joint_position_limits"]
        self.assertEqual(limits["fr_calf_joint"], (0., 3.14))
        self.assertEqual(limits["fl_calf_joint"], (-3.14, 0.))
        self.assertEqual(limits["rr_calf_joint"], (-3.14, 3.14))
        self.assertEqual(limits["rl_calf_joint"], (-3.14, 3.14))
        for leg in ("fr", "fl", "rr", "rl"):
            self.assertEqual(limits[f"{leg}_hip_joint"], (-.41, .41))

    def test_axes_and_positive_inertias(self):
        root = ET.parse(URDF).getroot()
        for joint in root.findall("joint"):
            if joint.attrib["type"] == "fixed":
                continue
            axis = [float(v) for v in joint.find("axis").attrib["xyz"].split()]
            self.assertAlmostEqual(sum(v*v for v in axis), 1., msg=joint.attrib["name"])
        for link in root.findall("link"):
            self.assertGreater(float(link.find("inertial/mass").attrib["value"]), 0.)
            inertia = link.find("inertial/inertia")
            self.assertIsNotNone(inertia, link.attrib["name"])
            a,b,c = (float(inertia.attrib[k]) for k in ("ixx", "iyy", "izz"))
            d,e,f = (float(inertia.attrib[k]) for k in ("ixy", "ixz", "iyz"))
            self.assertTrue(all(math.isfinite(v) for v in (a,b,c,d,e,f)))
            self.assertGreater(a, 0.)
            self.assertGreater(a*b-d*d, 0.)
            self.assertGreater(a*(b*c-f*f)-d*(d*c-e*f)+e*(d*f-b*e), 0.)

    def test_shipped_stl_payloads_are_complete(self):
        meshes = {mesh.attrib["filename"] for mesh in ET.parse(URDF).getroot().findall(".//mesh")}
        for mesh in meshes:
            path = URDF.parent / mesh
            with path.open("rb") as stream:
                header = stream.read(84)
            self.assertEqual(len(header), 84)
            triangles = struct.unpack("<I", header[80:84])[0]
            self.assertGreater(triangles, 0)
            self.assertEqual(path.stat().st_size, 84+50*triangles, mesh)

    def test_rejects_nonfinite_limits_duplicate_joints_and_continuous_leg(self):
        original = ET.parse(URDF).getroot()
        for mutation in ("nonfinite", "duplicate", "continuous_leg"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = copy.deepcopy(original)
                hip = root.find("./joint[@name='fr_hip_joint']")
                if mutation == "nonfinite":
                    hip.find("limit").set("upper", "inf")
                elif mutation == "duplicate":
                    root.append(copy.deepcopy(hip))
                else:
                    hip.set("type", "continuous")
                target = Path(directory) / "robot.urdf"
                ET.ElementTree(root).write(target)
                with self.assertRaises(ValueError):
                    validate_urdf(target, verify_meshes=False)

    def test_explicit_pose_rejects_missing_extra_nonfinite_and_out_of_range_positions(self):
        candidates = []
        missing = dict(POSE); missing.pop("fr_foot_joint"); candidates.append(missing)
        extra = dict(POSE, unexpected_joint=0.); candidates.append(extra)
        for name in ("fr_hip_joint", "fr_foot_joint"):
            for value in (float("nan"), float("inf"), -float("inf"), "0", True):
                candidates.append(dict(POSE, **{name: value}))
        candidates.append(dict(POSE, fr_hip_joint=.42))
        for pose in candidates:
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                validate_urdf(URDF, pose=pose)

    def test_explicit_pose_does_not_mutate_legacy_reference(self):
        before = dict(POSE)
        alternative = dict(POSE, fr_thigh_joint=-.7)
        validate_urdf(URDF, pose=alternative)
        self.assertEqual(POSE, before)
        self.assertEqual(POSE["fr_thigh_joint"], -.93)


if __name__ == "__main__":
    unittest.main()
