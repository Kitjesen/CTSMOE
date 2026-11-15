"""Validate the shipped Lope URDF without a simulator.

Feet are fixed joints. There is no wheel joint and no continuous joint.
The URDF ``effort`` attribute is the rejected flat 17 N·m cap; actuator limits
live in ``lope_lab3.spec`` and are applied by the Isaac Lab actuator config.
"""
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from lope_lab3.spec import (
    FOOT_JOINTS,
    FOOT_SPHERE_RADIUS_M,
    JOINT_NAMES,
    JOINT_POSITION_LIMITS_RAD,
    REJECTED_FLAT_EFFORT_NM,
    STANDING_POSE,
    URDF_SHA256,
)


def packaged_urdf():
    return Path(__file__).resolve().parent / "lope" / "urdf" / "lope.urdf"


def _finite(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Expected a finite number for {name}")
    return float(value)


def validate_urdf(path=None, *, verify_meshes=True):
    """Return a report for the 12 revolute joints and four fixed feet."""
    path = packaged_urdf() if path is None else Path(path)
    path = path.resolve(strict=True)
    root = ET.parse(path).getroot()
    if root.attrib.get("name") != "lope":
        raise ValueError(f"Expected robot name 'lope', got {root.attrib.get('name')!r}")
    joints = list(root.findall("joint"))
    by_name = {}
    for joint in joints:
        name = joint.attrib["name"]
        if name in by_name:
            raise ValueError(f"Duplicate joint: {name}")
        by_name[name] = joint
    moving = [joint for joint in joints if joint.attrib["type"] != "fixed"]
    moving_names = [joint.attrib["name"] for joint in moving]
    if moving_names != list(JOINT_NAMES):
        raise ValueError(f"Expected 12 revolute joints in canonical order, got {moving_names}")
    for joint in moving:
        if joint.attrib["type"] != "revolute":
            raise ValueError(f"Lope has no continuous or wheel joint: {joint.attrib['name']}")
        if joint.attrib["name"].endswith("_foot_joint"):
            raise ValueError("Foot joints must stay fixed")
    position_limits = {}
    axes = {}
    urdf_effort = {}
    for name in JOINT_NAMES:
        joint = by_name[name]
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Missing limit: {name}")
        lower = _finite(float(limit.attrib["lower"]), name)
        upper = _finite(float(limit.attrib["upper"]), name)
        effort = _finite(float(limit.attrib["effort"]), name)
        velocity = _finite(float(limit.attrib["velocity"]), name)
        if not lower < upper or effort <= 0 or velocity <= 0:
            raise ValueError(f"Invalid URDF limit: {name}")
        pose = STANDING_POSE[name]
        if not lower <= pose <= upper:
            raise ValueError(f"Standing pose outside URDF range: {name}={pose}")
        expected = JOINT_POSITION_LIMITS_RAD[name]
        if abs(lower - expected[0]) > 1e-9 or abs(upper - expected[1]) > 1e-9:
            raise ValueError(f"URDF position limit drifted from joint_motion_config: {name}")
        position_limits[name] = (lower, upper)
        axis = tuple(_finite(float(part), name) for part in joint.find("axis").attrib["xyz"].split())
        if len(axis) != 3 or abs(sum(part * part for part in axis) - 1.0) > 1e-6:
            raise ValueError(f"Revolute axis must be a unit vector: {name}")
        axes[name] = axis
        urdf_effort[name] = effort
        if effort != REJECTED_FLAT_EFFORT_NM:
            raise ValueError(
                "lope.urdf effort is expected to remain the rejected flat 17 N·m field; "
                "RS02 limits belong in the actuator config"
            )
    foot_radius = {}
    for name in FOOT_JOINTS:
        joint = by_name.get(name)
        if joint is None or joint.attrib["type"] != "fixed":
            raise ValueError(f"Expected a fixed foot joint: {name}")
        link = root.find(f"./link[@name='{name.replace('_joint', '_Link')}']")
        spheres = link.findall("./collision/geometry/sphere")
        if len(spheres) != 1:
            raise ValueError(f"Expected one foot contact sphere: {name}")
        radius = _finite(float(spheres[0].attrib["radius"]), name)
        if abs(radius - FOOT_SPHERE_RADIUS_M) > 1e-12:
            raise ValueError(f"Foot sphere radius must stay {FOOT_SPHERE_RADIUS_M} m")
        foot_radius[name] = radius
    links = {link.attrib["name"] for link in root.findall("link")}
    children = {joint.find("child").attrib["link"] for joint in joints}
    bases = links - children
    if bases != {"base_link"}:
        raise ValueError(f"Expected base_link as the only root, got {bases}")
    meshes = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if "://" in filename:
            raise ValueError("Resolve URDF mesh paths before import")
        resolved = (path.parent / filename).resolve()
        if verify_meshes and not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved not in meshes:
            meshes.append(resolved)
    total_mass = sum(float(link.find("inertial/mass").attrib["value"]) for link in root.findall("link"))
    if not math.isfinite(total_mass) or not 13.0 < total_mass < 14.0:
        raise ValueError(f"Unexpected Lope mass: {total_mass}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path == packaged_urdf().resolve() and digest != URDF_SHA256:
        raise ValueError("Packaged lope.urdf hash does not match the vendored asset commit")
    return {
        "urdf": str(path),
        "sha256": digest,
        "root_link": "base_link",
        "robot_name": "lope",
        "total_urdf_mass_kg": total_mass,
        "joint_names": list(JOINT_NAMES),
        "foot_joints": list(FOOT_JOINTS),
        "wheel_joint_count": 0,
        "continuous_joint_count": 0,
        "mesh_count": len(meshes),
        "meshes": meshes,
        "joint_position_limits": position_limits,
        "joint_axes": axes,
        "urdf_effort_nm": urdf_effort,
        "foot_sphere_radius_m": foot_radius,
        "standing_pose": dict(STANDING_POSE),
    }
