"""Static Lope checks. They do not start Isaac Lab or Newton."""
import math
import struct
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from lope_lab3.assets.contract import packaged_urdf, validate_urdf
from lope_lab3.spec import (
    ACTION_DIM,
    ACTION_SCALE_RAD,
    BASE_HEIGHT_TARGET_M,
    CUMULATIVE_UPDATES,
    EPISODE_LENGTH_S,
    JOINT_NAMES,
    JOINT_SIGN,
    LOPE_COMMAND_RANGE_CURRICULUM,
    LOPE_CLIP_RULES,
    LOPE_INITIAL_RANGES,
    LOPE_PUSH_LINEAR,
    LOPE_TERRAIN_MAX_COMMANDS,
    NUM_ENVS,
    OFFICIAL_COMMAND_RANGE_CURRICULUM,
    OFFICIAL_INITIAL_RANGES,
    RS02_CALF,
    RS02_HIP_THIGH,
    SPAWN_HEIGHT_M,
    STANDING_POSE,
    TERRAIN,
    TRACK_LIN_STD,
    TRACK_YAW_STD,
    action_clips,
)
from lope_lab3.spec import CRITIC_DIM, DECIMATION, FRAME_DIM, PHYSICS_DT_S, POLICY_DIM
from lope_lab3.recovery import FALLEN_MODES, fallen_orientation, recovering
from lope_lab3.spec import (
    FALLEN_RESET_FRACTION,
    FALLEN_TIMEOUT_S,
    RECOVERY_HEIGHT_WEIGHT,
    RECOVERY_UPRIGHT_WEIGHT,
    STAND_UP_BONUS_WEIGHT,
)
from lope_lab3.symmetry_math import (
    mirror_critic,
    mirror_joints,
    mirror_policy_frame,
    reflected_ray_indices,
    standing_vector,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source" / "lope_lab3"


class LopeAssetTests(unittest.TestCase):
    def test_urdf_is_twelve_revolute_joints_and_fixed_feet(self):
        report = validate_urdf()
        self.assertEqual(report["robot_name"], "lope")
        self.assertEqual(report["joint_names"], list(JOINT_NAMES))
        self.assertEqual(report["wheel_joint_count"], 0)
        self.assertEqual(report["continuous_joint_count"], 0)
        self.assertEqual(len(report["foot_joints"]), 4)
        self.assertEqual(report["root_link"], "base_link")
        self.assertAlmostEqual(report["total_urdf_mass_kg"], 13.333464609253825, places=6)
        self.assertGreaterEqual(report["mesh_count"], 35)
        for name, radius in report["foot_sphere_radius_m"].items():
            self.assertEqual(radius, 0.032, name)
        root = ET.parse(packaged_urdf()).getroot()
        moving = [joint.attrib["type"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]
        self.assertEqual(moving, ["revolute"] * 12)
        self.assertNotIn("continuous", {joint.attrib["type"] for joint in root.findall("joint")})

    def test_standing_pose_is_inside_hard_and_soft_limits(self):
        report = validate_urdf()
        clips = action_clips(report["joint_position_limits"])
        for name, pose in STANDING_POSE.items():
            lower, upper = report["joint_position_limits"][name]
            self.assertLessEqual(lower, pose)
            self.assertLessEqual(pose, upper)
            soft_lower, soft_upper = clips[name]
            self.assertLessEqual(soft_lower, pose)
            self.assertLessEqual(pose, soft_upper)
            self.assertLess(soft_lower, lower + (upper - lower) * 0.05 + 1e-9)
            self.assertGreater(soft_upper, upper - (upper - lower) * 0.05 - 1e-9)

    def test_urdf_axes_match_the_mirrored_left_right_convention(self):
        axes = validate_urdf()["joint_axes"]
        self.assertEqual(axes["fr_hip_joint"], (-1.0, 0.0, 0.0))
        self.assertEqual(axes["fl_hip_joint"], (-1.0, 0.0, 0.0))
        self.assertEqual(axes["rr_hip_joint"], (1.0, 0.0, 0.0))
        self.assertEqual(axes["rl_hip_joint"], (1.0, 0.0, 0.0))
        for leg in ("fr", "rr"):
            self.assertEqual(axes[f"{leg}_thigh_joint"], (0.0, 1.0, 0.0))
            self.assertEqual(axes[f"{leg}_calf_joint"], (0.0, 1.0, 0.0))
        for leg in ("fl", "rl"):
            self.assertEqual(axes[f"{leg}_thigh_joint"], (0.0, -1.0, 0.0))
            self.assertEqual(axes[f"{leg}_calf_joint"], (0.0, -1.0, 0.0))

    def test_referenced_stl_payloads_exist(self):
        report = validate_urdf()
        for mesh in report["meshes"]:
            if mesh.suffix.lower() != ".stl":
                continue
            payload = mesh.read_bytes()
            self.assertGreater(len(payload), 84, mesh.name)
            if payload[:5].lower() == b"solid":
                self.assertIn(b"facet", payload.lower(), mesh.name)
                continue
            triangles = struct.unpack("<I", payload[80:84])[0]
            self.assertGreater(triangles, 0, mesh.name)
            self.assertEqual(len(payload), 84 + 50 * triangles, mesh.name)

    def test_rs02_limits_are_not_the_flat_17nm_cap_or_thunder(self):
        self.assertEqual(RS02_HIP_THIGH["effort_limit"], 7.0)
        self.assertEqual(RS02_HIP_THIGH["saturation_effort"], 17.0)
        self.assertEqual(RS02_HIP_THIGH["velocity_limit"], 42.9350995991)
        self.assertEqual(RS02_HIP_THIGH["stiffness"], 40.0)
        self.assertEqual(RS02_HIP_THIGH["damping"], 0.8)
        self.assertEqual(RS02_HIP_THIGH["armature"], 0.0042)
        self.assertEqual(RS02_CALF["effort_limit"], 14.0)
        self.assertEqual(RS02_CALF["saturation_effort"], 34.0)
        self.assertEqual(RS02_CALF["velocity_limit"], 21.4675497995)
        self.assertEqual(RS02_CALF["stiffness"], 60.0)
        self.assertEqual(RS02_CALF["damping"], 1.5)
        self.assertEqual(RS02_CALF["armature"], 0.0168)
        self.assertNotEqual(RS02_HIP_THIGH["effort_limit"], 17.0)
        self.assertNotEqual(RS02_CALF["effort_limit"], 17.0)
        robot = (PACKAGE / "assets" / "robot.py").read_text(encoding="utf-8")
        self.assertIn("hip_thigh_rs02", robot)
        self.assertIn("calf_rs02_2to1", robot)
        self.assertIn("merge_fixed_joints=MERGE_FIXED_JOINTS", robot)
        self.assertIn("self_collision=SELF_COLLISION", robot)
        self.assertNotIn("ImplicitActuator", robot)
        self.assertNotIn("120", robot)
        self.assertNotIn("17.48", robot)
        self.assertNotIn("NEWPOSE_URDF", robot)


class LopeSymmetryTests(unittest.TestCase):
    def test_mirror_fixes_the_standing_pose_and_is_an_involution(self):
        pose = standing_vector()
        self.assertEqual(mirror_joints(pose), pose)
        sample = [0.2, -0.4, 0.8, 0.1, 0.3, -0.7, -0.2, 0.5, 1.1, 0.4, -0.6, -0.9]
        self.assertEqual(mirror_joints(mirror_joints(sample)), sample)
        self.assertEqual(JOINT_SIGN, (-1,) * 12)
        self.assertEqual(len(pose), ACTION_DIM)

    def test_policy_and_critic_mirrors_are_involutions(self):
        frame = [float((index % 7) - 3) for index in range(FRAME_DIM)]
        self.assertEqual(mirror_policy_frame(mirror_policy_frame(frame)), frame)
        xs = [index * 0.1 - 0.8 for index in range(17)]
        ys = [index * 0.1 - 0.5 for index in range(11)]
        starts = [(x, y, 20.0) for y in ys for x in xs]
        self.assertEqual(len(starts), 187)
        perm = reflected_ray_indices(starts)
        critic = [float(index) for index in range(CRITIC_DIM)]
        self.assertEqual(mirror_critic(mirror_critic(critic, perm), perm), critic)
        self.assertEqual(POLICY_DIM, 450)
        self.assertEqual(CRITIC_DIM, 263)


class LopeCurriculumTests(unittest.TestCase):
    def test_official_commands_are_unscaled_and_hardware_stays_locked(self):
        self.assertEqual(NUM_ENVS, 16384)
        self.assertEqual(CUMULATIVE_UPDATES, 100000)
        self.assertEqual(EPISODE_LENGTH_S, 25.0)
        self.assertNotEqual(BASE_HEIGHT_TARGET_M, 0.38)
        self.assertEqual(LOPE_INITIAL_RANGES, OFFICIAL_INITIAL_RANGES)
        self.assertEqual(LOPE_INITIAL_RANGES["lin_vel_x"], (-0.5, 0.5))
        self.assertEqual(LOPE_INITIAL_RANGES["lin_vel_y"], (-0.5, 0.5))
        self.assertEqual(LOPE_INITIAL_RANGES["ang_vel_yaw"], (-1.0, 1.0))
        self.assertEqual(LOPE_COMMAND_RANGE_CURRICULUM, OFFICIAL_COMMAND_RANGE_CURRICULUM)
        final = LOPE_COMMAND_RANGE_CURRICULUM[-1]
        self.assertEqual(final["iter"], 45000)
        self.assertEqual(final["lin_vel_x"], (-4.5, 4.2))
        self.assertEqual(final["lin_vel_y"], (-1.0, 1.0))
        self.assertEqual(final["ang_vel_yaw"], (-3.0, 3.0))
        self.assertEqual(LOPE_COMMAND_RANGE_CURRICULUM[0]["lin_vel_x"], (-1.0, 1.0))
        self.assertEqual(LOPE_COMMAND_RANGE_CURRICULUM[0]["ang_vel_yaw"], (-2.0, 2.0))
        self.assertNotIn("wheels", LOPE_TERRAIN_MAX_COMMANDS)
        self.assertEqual(LOPE_TERRAIN_MAX_COMMANDS["flat"]["lin_vel_x"], (-4.5, 4.2))
        self.assertEqual(LOPE_TERRAIN_MAX_COMMANDS["stairs_up"]["lin_vel_x"], (-1.0, 1.0))
        self.assertEqual(LOPE_PUSH_LINEAR["x"], (-0.4, 0.4))
        self.assertEqual(LOPE_CLIP_RULES[0]["limit"], 2.0)
        self.assertEqual(LOPE_CLIP_RULES[0]["clip"]["lin_vel_y"], (-0.5, 0.5))
        self.assertEqual(LOPE_CLIP_RULES[1]["limit"], 2.0)
        self.assertEqual(LOPE_CLIP_RULES[1]["clip"]["lin_vel_x"], (-2.0, 2.0))
        self.assertEqual(TRACK_LIN_STD, 0.5)
        self.assertEqual(TRACK_YAW_STD, 0.5)
        self.assertEqual(TERRAIN["stair_step_height_range"], (0.05, 0.25))
        self.assertEqual(TERRAIN["stair_step_width"], 0.25)
        self.assertEqual(TERRAIN["num_rows"], 10)
        self.assertEqual(TERRAIN["max_init_terrain_level"], 5)
        self.assertEqual(TERRAIN["slope_range"][0], 0.0)
        self.assertAlmostEqual(TERRAIN["slope_range"][1], math.tan(math.radians(35)))
        self.assertNotAlmostEqual(TERRAIN["slope_range"][1], math.radians(35))
        self.assertNotAlmostEqual(TERRAIN["slope_range"][1], math.tan(math.radians(40)))
        self.assertNotAlmostEqual(TERRAIN["slope_range"][1], 0.568)
        self.assertEqual(PHYSICS_DT_S, 0.005)
        self.assertEqual(DECIMATION, 4)
        self.assertEqual(SPAWN_HEIGHT_M, 0.2740510271690917)
        self.assertEqual(BASE_HEIGHT_TARGET_M, 0.2627175562793766)
        self.assertEqual(STANDING_POSE["fr_hip_joint"], -0.08)
        self.assertEqual(STANDING_POSE["fl_hip_joint"], 0.08)
        self.assertEqual(STANDING_POSE["rr_hip_joint"], 0.08)
        self.assertEqual(STANDING_POSE["rl_hip_joint"], -0.08)
        self.assertEqual(STANDING_POSE["fr_calf_joint"], 1.1)
        self.assertEqual(STANDING_POSE["fr_thigh_joint"], -0.6715534335840251)
        self.assertEqual(ACTION_SCALE_RAD, {"hip": 0.15, "thigh": 0.2, "calf": 0.2})
        pose = standing_vector()
        self.assertEqual(mirror_joints(pose), pose)
        banned_scale = ("0.6741009210204774", "0.3370504605102387", "-3.033454144592148", "2.831223868286005")
        for path in PACKAGE.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in banned_scale:
                self.assertNotIn(token, text, path.name)

    def test_recovery_is_in_the_same_task_without_instant_base_contact(self):
        self.assertEqual(FALLEN_RESET_FRACTION, 0.20)
        self.assertEqual(FALLEN_TIMEOUT_S, 6.0)
        self.assertEqual(RECOVERY_UPRIGHT_WEIGHT, 2.0)
        self.assertEqual(RECOVERY_HEIGHT_WEIGHT, -2.0)
        self.assertEqual(STAND_UP_BONUS_WEIGHT, 5.0)
        for mode in FALLEN_MODES:
            roll, pitch, _yaw = fallen_orientation(mode, 0.0, 0.0, 0.2)
            self.assertGreater(max(abs(roll), abs(pitch)), 1.0, mode)
            self.assertTrue(recovering(math.sin(roll), 0.9) or abs(roll) > 1.0 or abs(pitch) > 1.0)
        env = (PACKAGE / "tasks" / "lope" / "env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("reset_root_with_recovery", env)
        self.assertIn("recovery_upright", env)
        self.assertIn("recovery_base_height", env)
        self.assertIn("stand_up_bonus", env)
        self.assertIn("fallen_timeout", env)
        self.assertIn("fallen_reset_fraction", env)
        self.assertNotIn("illegal_contact", env)
        self.assertNotIn("bad_orientation", env)

    def test_registration_names_the_lope_task(self):
        text = (PACKAGE / "tasks" / "lope" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("gym.register", text)
        self.assertIn("LopeFreshEnvCfg", text)
        self.assertIn("LopeFreshPlayEnvCfg", text)
        self.assertIn("LopeFreshMoeCtsRunnerCfg", text)
        self.assertIn("TASK_ID", text)
        self.assertNotIn("thunder_lab3", text)
        self.assertNotIn("Thunder-Newpose", text)
        robot = (PACKAGE / "assets" / "robot.py").read_text(encoding="utf-8")
        env = (PACKAGE / "tasks" / "lope" / "env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("BASE_HEIGHT_TARGET_M", env)
        self.assertIn("SPAWN_HEIGHT_M", env)
        self.assertNotIn("joint_vel", env)
        self.assertIn("RS02_HIP_THIGH", robot)
        self.assertIn("RS02_CALF", robot)
        scene = (PACKAGE / "tasks" / "lope" / "scene_cfg.py").read_text(encoding="utf-8")
        self.assertIn('prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link"', scene)
        self.assertNotIn('prim_path="{ENV_REGEX_NS}/Robot/base_link"', scene)

    def test_task_sources_do_not_quote_thunder_targets(self):
        banned = ("0.391415", "0.411415", "A_POSE", "ImplicitActuator", "lin_vel_x=(-4.5", "radians(40)")
        for path in PACKAGE.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, text, f"{path.name} contains {token}")


if __name__ == "__main__":
    unittest.main()
