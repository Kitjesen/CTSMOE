"""Backend-neutral checks for the Thunder asset and its unchanged reference pose."""
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

POSE = {
    "fr_hip_joint": -0.12, "fr_thigh_joint": -0.93, "fr_calf_joint": 1.96,
    "fl_hip_joint": 0.12, "fl_thigh_joint": 0.93, "fl_calf_joint": -1.96,
    "rr_hip_joint": 0.12, "rr_thigh_joint": 2.1501, "rr_calf_joint": 1.8405,
    "rl_hip_joint": -0.12, "rl_thigh_joint": -2.1501, "rl_calf_joint": -1.8405,
    "fr_foot_joint": 0.0, "fl_foot_joint": 0.0, "rr_foot_joint": 0.0, "rl_foot_joint": 0.0,
}


def validate_urdf(path, *, verify_meshes=True):
    path = Path(path).resolve(strict=True)
    root = ET.parse(path).getroot()
    moving = [j for j in root.findall("joint") if j.attrib["type"] != "fixed"]
    joints = {j.attrib["name"]: j for j in moving}
    if len(moving) != len(POSE) or set(joints) != set(POSE):
        raise ValueError(f"Expected precisely 16 NEWPOSE joints; got {sorted(joints)}")
    position_limits = {}
    for name, q in POSE.items():
        j = joints[name]
        expected_type = "continuous" if name.endswith("_foot_joint") else "revolute"
        if j.attrib["type"] != expected_type:
            raise ValueError(f"Expected {expected_type} joint: {name}")
        lim = j.find("limit")
        if lim is None or any(not math.isfinite(float(lim.attrib[key])) or float(lim.attrib[key]) <= 0
                              for key in ("effort", "velocity")):
            raise ValueError(f"Expected finite positive effort/velocity limits: {name}")
        if expected_type == "revolute":
            lower, upper = float(lim.attrib["lower"]), float(lim.attrib["upper"])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper or not lower <= q <= upper:
                raise ValueError(f"Default pose violates URDF range: {name}={q}")
            position_limits[name] = (lower, upper)
    links = {v.attrib["name"] for v in root.findall("link")}
    children = {j.find("child").attrib["link"] for j in root.findall("joint")}
    bases = links - children
    if len(bases) != 1:
        raise ValueError(f"Expected one root link, got {bases}")
    meshes = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if "://" in filename:
            raise ValueError("Resolve URDF mesh URIs before this standalone import smoke")
        resolved = (path.parent / filename).resolve()
        if verify_meshes and not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved not in meshes:
            meshes.append(resolved)
    total_mass = sum(float(m.attrib["value"]) for m in root.findall("./link/inertial/mass"))
    if not math.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Invalid total URDF mass")
    return {"urdf": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "root_link": next(iter(bases)), "total_urdf_mass_kg": total_mass,
            "joint_names": list(joints), "mesh_count": len(meshes),
            "joint_position_limits": position_limits}
