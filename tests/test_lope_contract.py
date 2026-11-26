"""Static Lope checks. They do not start Isaac Lab or Newton."""
import ast
import math
import struct
import sys
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from lope_lab3.assets.contract import packaged_urdf, validate_urdf
from lope_lab3.spec import (
    ACTION_DIM,
    ACTION_SCALE_RAD,
    BAD_ORIENTATION_LIMIT_RAD,
    BASE_HEIGHT_TARGET_M,
    BODY_COM_OFFSET_X,
    CUMULATIVE_UPDATES,
    EPISODE_LENGTH_S,
    FOOT_BODIES,
    ILLEGAL_CONTACT_THRESHOLD_N,
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
    RS02_BATTERY_CELLS,
    RS02_CALF,
    RS02_HIP_THIGH,
    SELF_COLLISION,
    SOLVER_NJMAX,
    SPAWN_HEIGHT_M,
    STANDING_POSE,
    TERRAIN,
    TRACK_LIN_STD,
    TRACK_YAW_STD,
    URDF_SHA256,
    action_clips,
)
from lope_lab3.spec import (
    CRITIC_DIM,
    CRITIC_PROPRIO_DIM,
    DECIMATION,
    FOOT_FORCE_DIM,
    FRAME_DIM,
    PHYSICS_DT_S,
    POLICY_DIM,
)
from lope_lab3.recovery import FALLEN_MODES, fallen_orientation, recovering, upright_deficit
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

    def test_cleaned_urdf_collisions_prevent_phantom_self_contact(self):
        report = validate_urdf()
        self.assertEqual(report["sha256"], URDF_SHA256)
        root = ET.parse(packaged_urdf()).getroot()
        links = {link.attrib["name"]: link for link in root.findall("link")}
        for internal_link in ("battery_guide_link", "battery_pack_trial_link"):
            self.assertEqual(links[internal_link].findall("collision"), [], internal_link)
        base_cols = {col.attrib["name"]: col for col in links["base_link"].findall("collision")}
        self.assertEqual(set(base_cols), {"body", "lope_upper"})
        self.assertEqual(base_cols["body"].find("geometry/box").attrib["size"], "0.26 0.19 0.12")
        self.assertEqual(base_cols["lope_upper"].find("geometry/mesh").attrib.get("scale"), "0.95 0.84 1.0")
        for leg in ("fr", "fl", "rr", "rl"):
            hip_cols = {col.attrib["name"]: col for col in links[f"{leg}_hip_Link"].findall("collision")}
            cyl = hip_cols["hip_housing"].find("geometry/cylinder")
            self.assertEqual(cyl.attrib["radius"], "0.036", leg)
            self.assertEqual(cyl.attrib["length"], "0.040", leg)
            thigh_cols = {col.attrib["name"]: col for col in links[f"{leg}_thigh_Link"].findall("collision")}
            self.assertEqual(set(thigh_cols), {"thigh_distal", "thigh_proximal"}, leg)

    def test_rs02_limits_are_not_the_flat_17nm_cap_or_thunder(self):
        self.assertEqual(RS02_BATTERY_CELLS, 14)
        self.assertEqual(RS02_HIP_THIGH["peak_torque_motoring"], 14.5)
        self.assertEqual(RS02_HIP_THIGH["peak_torque_braking"], 16.5)
        self.assertEqual(RS02_HIP_THIGH["continuous_torque_nm"], 7.0)
        self.assertEqual(RS02_HIP_THIGH["corner_velocity"], 24.0)
        self.assertEqual(RS02_HIP_THIGH["velocity_limit"], 42.9350995991)
        self.assertEqual(RS02_HIP_THIGH["stiffness"], 40.0)
        self.assertEqual(RS02_HIP_THIGH["damping"], 0.8)
        self.assertEqual(RS02_HIP_THIGH["armature"], 0.0042)
        self.assertEqual(RS02_HIP_THIGH["friction"], 0.01)
        self.assertEqual(RS02_HIP_THIGH["min_delay"], 0)
        self.assertEqual(RS02_HIP_THIGH["max_delay"], 3)
        self.assertEqual(RS02_CALF["peak_torque_motoring"], 28.0)
        self.assertEqual(RS02_CALF["peak_torque_braking"], 32.0)
        self.assertEqual(RS02_CALF["continuous_torque_nm"], 14.0)
        self.assertEqual(RS02_CALF["corner_velocity"], 12.0)
        self.assertEqual(RS02_CALF["velocity_limit"], 21.4675497995)
        self.assertEqual(RS02_CALF["stiffness"], 60.0)
        self.assertEqual(RS02_CALF["damping"], 1.5)
        self.assertEqual(RS02_CALF["armature"], 0.0168)
        self.assertEqual(RS02_CALF["friction"], 0.02)
        self.assertEqual(RS02_CALF["min_delay"], 0)
        self.assertEqual(RS02_CALF["max_delay"], 3)
        self.assertNotEqual(RS02_HIP_THIGH["peak_torque_motoring"], 17.0)
        self.assertNotEqual(RS02_CALF["peak_torque_motoring"], 17.0)
        robot = (PACKAGE / "assets" / "robot.py").read_text(encoding="utf-8")
        self.assertIn("RS02ActuatorCfg", robot)
        self.assertNotIn("DCMotorCfg", robot)
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
        once = mirror_critic(critic, perm)
        self.assertEqual(once[72:76], [73.0, 72.0, 75.0, 74.0])
        self.assertEqual(once[76:80], [77.0, 76.0, 79.0, 78.0])
        self.assertEqual(once[80:84], [81.0, 80.0, 83.0, 82.0])
        self.assertEqual(once[84:88], [85.0, 84.0, 87.0, 86.0])
        self.assertEqual(POLICY_DIM, 450)
        self.assertEqual(FOOT_FORCE_DIM, 16)
        self.assertEqual(CRITIC_PROPRIO_DIM, 88)
        self.assertEqual(CRITIC_DIM, 275)


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
        self.assertEqual(TERRAIN["max_init_terrain_level"], 1)
        self.assertTrue(SELF_COLLISION)
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

    def test_locomotion_task_is_pure_and_aligned_with_go2(self):
        self.assertEqual(ILLEGAL_CONTACT_THRESHOLD_N, 1.0)
        self.assertEqual(BAD_ORIENTATION_LIMIT_RAD, 2.0)
        self.assertEqual(upright_deficit(-1.0, 0.9), 0.0)
        self.assertAlmostEqual(upright_deficit(-0.8, 0.9), 0.0, places=9)
        self.assertGreater(upright_deficit(-0.79, 0.9), 0.0)
        self.assertLess(upright_deficit(-0.79, 0.9), 0.02)
        self.assertAlmostEqual(upright_deficit(0.0, 0.9), 0.4, places=9)
        self.assertAlmostEqual(upright_deficit(1.0, 0.9), 0.9, places=9)
        for mode in FALLEN_MODES:
            roll, pitch, _yaw = fallen_orientation(mode, 0.0, 0.0, 0.2)
            self.assertGreater(max(abs(roll), abs(pitch)), 1.0, mode)
            self.assertTrue(recovering(math.sin(roll), 0.9) or abs(roll) > 1.0 or abs(pitch) > 1.0)
        env = (PACKAGE / "tasks" / "lope" / "env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("reset_root_state_uniform", env)
        self.assertIn("go2_rewards.undesired_contacts", env)
        self.assertIn("lab_mdp.illegal_contact", env)
        self.assertIn("lab_mdp.bad_orientation", env)
        self.assertIn("self.sim.physics_material = self.scene.terrain.physics_material", env)
        for forbidden in (
            "reset_root_with_recovery",
            "recovery_upright",
            "stand_up_bonus",
            "fallen_timeout",
            "fallen_reset_fraction",
            "undesired_contacts_locomotion",
            "bad_orientation_locomotion",
        ):
            self.assertNotIn(forbidden, env)
        fresh_state = (PACKAGE / "training" / "fresh_state.py").read_text(encoding="utf-8")
        self.assertIn('"assets/actuator.py"', fresh_state)

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
        self.assertIn(
            "from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg",
            scene,
        )
        self.assertIn("contact_forces = NewtonContactSensorCfg(", scene)
        self.assertIn("history_length=3", scene)
        self.assertIn("track_air_time=True", scene)
        self.assertNotIn("from isaaclab.sensors import ContactSensorCfg", scene)

    def test_noisy_command_x_scale_is_one_half(self):
        source = (PACKAGE / "tasks" / "lope" / "mdp" / "observations.py").read_text(encoding="utf-8")
        self.assertIn(
            "[0.2] * 3 + [0.05] * 3 + [0.0] * 3 + [0.03] * 12 + [2.0] * 12 + [0.0] * 12",
            source,
        )
        scale_expr = "[0.25] * 3 + [1.0] * 3 + [0.5, 1.0, 1.0] + [1.0] * 12 + [0.05] * 12 + [1.0] * 12"
        self.assertIn(scale_expr, source)
        self.assertNotIn("[1.0] * 18", source)
        scale = eval(scale_expr, {"__builtins__": {}})
        self.assertEqual(len(scale), FRAME_DIM)
        command = [2.0, -3.0, 4.0]
        scaled = [value * factor for value, factor in zip(command, scale[6:9])]
        self.assertEqual(scaled[0], 1.0)
        self.assertEqual(scaled[1:], [-3.0, 4.0])

    def test_solver_njmax_and_com_buckets(self):
        self.assertEqual(SOLVER_NJMAX, 512)
        self.assertEqual(BODY_COM_OFFSET_X, (-0.05, 0.05))
        text = (PACKAGE / "tasks" / "lope" / "mdp" / "randomization.py").read_text(encoding="utf-8")
        self.assertIn("njmax=SOLVER_NJMAX", text)
        self.assertNotIn("njmax=300", text)
        self.assertIn("nconmax=128", text)

    def test_entropy_coef_is_tempered(self):
        text = (PACKAGE / "tasks" / "lope" / "rsl_rl_cfg.py").read_text(encoding="utf-8")
        self.assertIn("entropy_coef=0.005", text)

    def test_critic_height_scan_keeps_nonfinite_values(self):
        path = PACKAGE / "tasks" / "lope" / "mdp" / "observations.py"
        source = path.read_text(encoding="utf-8")
        self.assertNotIn("ray_hits_w", source)
        self.assertNotIn("net_normal_forces_w", source)
        self.assertIn("from go2_rl_lab.tasks.mixed.mdp.observations import foot_contact_force_norm", source)
        self.assertIn("foot_contact_force_norm(env, sensor_cfg)", source)
        self.assertNotIn("pos_w.torch[:, 2:3]", source)
        self.assertIn(
            'ray_caster_height_scan(env, SceneEntityCfg("height_scanner"), offset=0.5).clamp(-1, 1) * 2.5',
            source,
        )
        function = next(
            node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "critic_frame"
        )
        self.assertFalse(any(isinstance(node, ast.Name) and node.id == "isfinite" for node in ast.walk(function)))
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == "isfinite" for node in ast.walk(function)))
        self.assertIn("result.shape != (env.num_envs, CRITIC_DIM)", source)

        critic_frame = _load_critic_frame_with_stubs()
        env, calls, height = _critic_env(width=CRITIC_DIM - CRITIC_PROPRIO_DIM, nonfinite=True)
        result = critic_frame(env)
        self.assertEqual(calls, [("height_scanner", 0.5)])
        self.assertEqual(_FORCE_CALLS, [(1, 0, 3, 2)])
        self.assertEqual(result.shape, (env.num_envs, CRITIC_DIM))
        self.assertTrue(any(math.isnan(value) for value in result.values[0]))
        height.values = [[0.0] * (CRITIC_DIM - CRITIC_PROPRIO_DIM - 1) for _ in range(env.num_envs)]
        with self.assertRaises(RuntimeError) as caught:
            critic_frame(env)
        self.assertNotIn("finite", str(caught.exception).lower())

    def test_task_sources_do_not_quote_thunder_targets(self):
        banned = ("0.391415", "0.411415", "A_POSE", "ImplicitActuator", "lin_vel_x=(-4.5", "radians(40)")
        for path in PACKAGE.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, text, f"{path.name} contains {token}")


def _sequence(value):
    return isinstance(value, (list, tuple))


class _Tensor:
    """Just enough of a tensor for critic_frame when PyTorch is not installed."""

    def __init__(self, values):
        self.values = values

    @property
    def shape(self):
        return self._shape(self.values)

    @staticmethod
    def _shape(values):
        if not _sequence(values):
            return ()
        return (len(values),) + _Tensor._shape(values[0])

    def new_tensor(self, values):
        return _Tensor(list(values))

    def clamp(self, low, high):
        def clip(value):
            if isinstance(value, float) and math.isnan(value):
                return value
            return min(high, max(low, value))

        return _Tensor(self._map(self.values, clip))

    def _map(self, values, function):
        if _sequence(values):
            return [self._map(item, function) for item in values]
        return function(values)

    def __mul__(self, other):
        if isinstance(other, _Tensor):
            return _Tensor(self._binop(self.values, other.values, lambda left, right: left * right))
        return _Tensor(self._map(self.values, lambda value: value * other))

    def __sub__(self, other):
        return _Tensor(self._binop(self.values, other.values, lambda left, right: left - right))

    def _binop(self, left, right, function):
        if not _sequence(left) and not _sequence(right):
            return function(left, right)
        if _sequence(left) and _sequence(right) and len(left) == len(right):
            return [self._binop(item, other, function) for item, other in zip(left, right)]
        if _sequence(left) and _sequence(right) and _sequence(left[0]) and len(right) == len(left[0]):
            return [self._binop(row, right, function) for row in left]
        raise AssertionError("critic stub cannot broadcast these tensors")

    def __getitem__(self, index):
        if isinstance(index, tuple) and index and index[0] == slice(None):
            if len(index) == 2:
                return _Tensor([[row[column] for column in index[1]] for row in self.values])
            if len(index) == 3 and index[1] == slice(None):
                selected = index[2]
                return _Tensor([[[step[column] for column in selected] for step in env] for env in self.values])
        raise AssertionError("critic stub received an unexpected index")


_HEIGHT_CALLS = []
_HEIGHT_BOX = {}
_FORCE_CALLS = []


def _ray_caster_height_scan(env, sensor_cfg, offset=0.5):
    del env
    _HEIGHT_CALLS.append((sensor_cfg.name, offset))
    return _HEIGHT_BOX["height"]


def _foot_contact_force_norm(env, sensor_cfg):
    """Official layout: max over history, then each history step, feet in body_ids order."""
    history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history.torch[:, :, list(sensor_cfg.body_ids)]
    _FORCE_CALLS.append(tuple(sensor_cfg.body_ids))
    rows = []
    for steps in history.values:
        step_norms = [[math.sqrt(sum(part * part for part in vector)) for vector in step] for step in steps]
        maximum = [max(step[foot] for step in step_norms) for foot in range(len(step_norms[0]))]
        rows.append(maximum + [value for step in step_norms for value in step])
    return _Tensor(rows)


def _load_critic_frame_with_stubs():
    saved = {name: sys.modules[name] for name in list(sys.modules) if name == "torch" or name.startswith(("isaaclab", "go2_rl_lab", "gymnasium"))}
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _Tensor
    torch_mod.cat = lambda parts, dim=-1: _cat(parts, dim)
    torch_mod.linalg = types.SimpleNamespace(vector_norm=_vector_norm)
    managers = types.ModuleType("isaaclab.managers")

    class ManagerTermBase:
        pass

    class SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    managers.ManagerTermBase = ManagerTermBase
    managers.SceneEntityCfg = SceneEntityCfg
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.managers = managers
    height_scan = types.ModuleType("go2_rl_lab.utils.height_scan")
    height_scan.ray_caster_height_scan = _ray_caster_height_scan
    go2_utils = types.ModuleType("go2_rl_lab.utils")
    go2_utils.height_scan = height_scan
    go2 = types.ModuleType("go2_rl_lab")
    go2.utils = go2_utils
    go2_tasks = types.ModuleType("go2_rl_lab.tasks")
    go2_mixed = types.ModuleType("go2_rl_lab.tasks.mixed")
    go2_mdp = types.ModuleType("go2_rl_lab.tasks.mixed.mdp")
    go2_obs = types.ModuleType("go2_rl_lab.tasks.mixed.mdp.observations")
    go2_obs.foot_contact_force_norm = _foot_contact_force_norm
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.register = lambda **kwargs: None
    installed = {
        "torch": torch_mod,
        "isaaclab": isaaclab,
        "isaaclab.managers": managers,
        "go2_rl_lab": go2,
        "go2_rl_lab.utils": go2_utils,
        "go2_rl_lab.utils.height_scan": height_scan,
        "go2_rl_lab.tasks": go2_tasks,
        "go2_rl_lab.tasks.mixed": go2_mixed,
        "go2_rl_lab.tasks.mixed.mdp": go2_mdp,
        "go2_rl_lab.tasks.mixed.mdp.observations": go2_obs,
        "gymnasium": gymnasium,
    }
    sys.modules.update(installed)
    for name in list(sys.modules):
        if name == "lope_lab3.tasks.lope" or name.startswith("lope_lab3.tasks.lope."):
            del sys.modules[name]
    try:
        from lope_lab3.tasks.lope.mdp.observations import critic_frame
    finally:
        for name in installed:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return critic_frame


def _cat(parts, dim):
    if dim != -1:
        raise AssertionError("critic stub only concatenates the last dimension")
    rows = []
    for index in range(parts[0].shape[0]):
        row = []
        for part in parts:
            item = part.values[index]
            row.extend(item if _sequence(item) else [item])
        rows.append(row)
    return _Tensor(rows)


def _vector_norm(tensor, dim):
    if dim != -1:
        raise AssertionError("critic stub only norms the last dimension")
    return _Tensor(
        [[math.sqrt(sum(component * component for component in vector)) for vector in env] for env in tensor.values]
    )


def _critic_env(width, nonfinite):
    count = 2
    _HEIGHT_CALLS.clear()
    _FORCE_CALLS.clear()
    scan = [[float("nan")] + [0.0] * (width - 1) for _ in range(count)] if nonfinite else [[0.0] * width for _ in range(count)]
    height = _Tensor(scan)
    _HEIGHT_BOX["height"] = height
    zeros = lambda columns: _Tensor([[0.0] * columns for _ in range(count)])
    vectors = zeros(3)
    joints = zeros(12)
    robot = types.SimpleNamespace(
        joint_names=list(JOINT_NAMES),
        data=types.SimpleNamespace(
            root_lin_vel_b=types.SimpleNamespace(torch=vectors),
            root_ang_vel_b=types.SimpleNamespace(torch=vectors),
            projected_gravity_b=types.SimpleNamespace(torch=vectors),
            joint_pos=types.SimpleNamespace(torch=joints),
            default_joint_pos=types.SimpleNamespace(torch=joints),
            joint_vel=types.SimpleNamespace(torch=joints),
            default_joint_vel=types.SimpleNamespace(torch=joints),
            joint_acc=types.SimpleNamespace(torch=joints),
        ),
        actuators=types.SimpleNamespace(applied_effort=types.SimpleNamespace(torch=joints)),
    )
    # Go2 USD order is FL, FR, RL, RR. Lope must resolve FR, FL, RR, RL itself.
    go2_order = (FOOT_BODIES[1], FOOT_BODIES[0], FOOT_BODIES[3], FOOT_BODIES[2])
    history = [[[[0.0, 0.0, 0.0] for _ in go2_order] for _ in range(3)] for _ in range(count)]
    sensor = types.SimpleNamespace(
        body_names=list(go2_order),
        data=types.SimpleNamespace(net_forces_w_history=types.SimpleNamespace(torch=_Tensor(history))),
    )

    class _Scene(dict):
        pass

    scene = _Scene(robot=robot, contact_forces=sensor)
    scene.sensors = {"contact_forces": sensor}
    env = types.SimpleNamespace(
        num_envs=count,
        scene=scene,
        command_manager=types.SimpleNamespace(get_command=lambda name: zeros(3)),
        action_manager=types.SimpleNamespace(action=joints),
    )
    return env, _HEIGHT_CALLS, height


if __name__ == "__main__":
    unittest.main()
