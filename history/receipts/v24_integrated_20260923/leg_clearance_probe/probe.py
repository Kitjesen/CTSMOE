"""Read-only URDF collision-envelope distances for fixed assessment traces.

This is a kinematic diagnostic, not a PhysX contact or hardware measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pybullet


HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE.parent.parent / "wave_reset_20260919" / "thunder_v4.urdf"
EXPECTED_URDF_SHA256 = (
    "83ba02ca52863238ce4050fff7c122fb159e381fb106ba31e39e44bfc0b66af3"
)
SIDES = ("fr", "fl", "rr", "rl")
CATEGORIES = ("inter_leg", "leg_body", "leg_body_distal")
PAIR_CATEGORIES = ("inter_leg", "leg_body_hip", "leg_body_distal")


@dataclass
class PairIndex:
    geometry_a: np.ndarray
    geometry_b: np.ndarray
    links: list[tuple[str, str]]


class BulletProximity:
    """Bullet is a geometry query engine only; no simulation step is executed."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.client = pybullet.connect(pybullet.DIRECT)
        self.bodies = []
        try:
            for index in range(model.ngeom):
                shape_type = model.geom_type[index]
                size = model.geom_size[index]
                if shape_type == mujoco.mjtGeom.mjGEOM_BOX:
                    shape = pybullet.createCollisionShape(
                        pybullet.GEOM_BOX,
                        halfExtents=size[:3].tolist(),
                        physicsClientId=self.client,
                    )
                elif shape_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    shape = pybullet.createCollisionShape(
                        pybullet.GEOM_CYLINDER,
                        radius=float(size[0]),
                        height=float(size[1] * 2),
                        physicsClientId=self.client,
                    )
                else:
                    raise ValueError(f"Unsupported MuJoCo shape type {shape_type}")
                self.bodies.append(
                    pybullet.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=shape,
                        physicsClientId=self.client,
                    )
                )
        except Exception:
            self.close()
            raise

    def update(self, data: mujoco.MjData) -> None:
        for index, body in enumerate(self.bodies):
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, data.geom_xmat[index])
            pybullet.resetBasePositionAndOrientation(
                body,
                data.geom_xpos[index],
                (quaternion[1], quaternion[2], quaternion[3], quaternion[0]),
                physicsClientId=self.client,
            )

    def distance(self, geometry_a: int, geometry_b: int) -> float:
        points = pybullet.getClosestPoints(
            self.bodies[geometry_a],
            self.bodies[geometry_b],
            10.0,
            physicsClientId=self.client,
        )
        if not points:
            raise ValueError(
                "Bullet did not return a closest point for a Thunder geometry pair"
            )
        return float(min(point[8] for point in points))

    def close(self) -> None:
        if pybullet.isConnected(self.client):
            pybullet.disconnect(self.client)


def _numbers(value: str | None, fallback: tuple[float, ...]) -> tuple[float, ...]:
    return fallback if value is None else tuple(float(part) for part in value.split())


def _quat(rpy: tuple[float, ...]) -> str:
    roll, pitch, yaw = (angle / 2 for angle in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    values = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    return " ".join(str(value) for value in values)


def _origin(element: ET.Element | None) -> dict[str, str]:
    xyz = _numbers(element.get("xyz") if element is not None else None, (0, 0, 0))
    rpy = _numbers(element.get("rpy") if element is not None else None, (0, 0, 0))
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError("URDF origin must have three xyz and three rpy values")
    return {"pos": " ".join(str(value) for value in xyz), "quat": _quat(rpy)}


def _leg_and_segment(link: str) -> tuple[str | None, str | None]:
    if link == "base_link":
        return None, "body"
    side, _, remainder = link.partition("_")
    if side not in SIDES:
        return None, None
    return side, remainder.lower().split("_")[0]


def build_model(urdf_path: Path) -> tuple[mujoco.MjModel, dict[str, PairIndex]]:
    """Import collision primitives and joint tree into a fixed-base MuJoCo model.

    Fixed base is valid here because pair distances are invariant to global pose.
    """
    robot = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = robot.findall("joint")
    children: dict[str, list[ET.Element]] = {name: [] for name in links}
    for joint in joints:
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent not in links or child not in links:
            raise ValueError(f"Unknown URDF link in joint {joint.get('name')}")
        children[parent].append(joint)

    root = ET.Element("mujoco", model="thunder_clearance_probe")
    ET.SubElement(root, "compiler", angle="radian", fusestatic="false")
    ET.SubElement(root, "option", gravity="0 0 0")
    world = ET.SubElement(root, "worldbody")
    geometries: dict[str, list[int]] = {}
    geom_names: dict[str, list[str]] = {}

    def add_link(parent: ET.Element, link_name: str, transform: dict[str, str]) -> None:
        body = ET.SubElement(parent, "body", name=link_name, **transform)
        ET.SubElement(
            body, "inertial", pos="0 0 0", mass="1", diaginertia="0.01 0.01 0.01"
        )
        geom_names[link_name] = []
        for index, collision in enumerate(links[link_name].findall("collision")):
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError(f"Malformed collision geometry on {link_name}")
            shape = geometry[0]
            if shape.tag == "box":
                dimensions = _numbers(shape.get("size"), ())
                if len(dimensions) != 3:
                    raise ValueError(f"Malformed box on {link_name}")
                size = tuple(value / 2 for value in dimensions)
            elif shape.tag == "cylinder":
                size = (float(shape.get("radius")), float(shape.get("length")) / 2)
            else:
                raise ValueError(f"Unsupported collision {shape.tag} on {link_name}")
            if not all(math.isfinite(value) and value > 0 for value in size):
                raise ValueError(f"Invalid collision dimensions on {link_name}")
            name = f"{link_name}__collision_{index}"
            ET.SubElement(
                body,
                "geom",
                name=name,
                type=shape.tag,
                size=" ".join(str(value) for value in size),
                contype="0",
                conaffinity="0",
                **_origin(collision.find("origin")),
            )
            geom_names[link_name].append(name)
        for joint in children[link_name]:
            child_name = joint.find("child").get("link")
            add_link(body, child_name, _origin(joint.find("origin")))
            child_body = body[-1]
            joint_type = joint.get("type")
            if joint_type in ("revolute", "continuous"):
                axis = _numbers(joint.find("axis").get("xyz"), ())
                if len(axis) != 3 or not any(axis):
                    raise ValueError(f"Invalid joint axis for {joint.get('name')}")
                ET.SubElement(
                    child_body,
                    "joint",
                    name=joint.get("name"),
                    type="hinge",
                    axis=" ".join(str(value) for value in axis),
                    limited="false",
                )
            elif joint_type != "fixed":
                raise ValueError(f"Unsupported joint type {joint_type}")

    add_link(world, "base_link", {"pos": "0 0 0", "quat": "1 0 0 0"})
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for link, names in geom_names.items():
        geometries[link] = [model.geom(name).id for name in names]
    pairs: list[tuple[str, int, int, str, str]] = []
    for i, left in enumerate(geometries):
        left_side, left_segment = _leg_and_segment(left)
        for right in list(geometries)[i + 1 :]:
            right_side, right_segment = _leg_and_segment(right)
            if (
                left_side is not None
                and right_side is not None
                and left_side != right_side
            ):
                category = "inter_leg"
            elif left_segment == "body" and right_side is not None:
                category = (
                    "leg_body_distal" if right_segment != "hip" else "leg_body_hip"
                )
            elif right_segment == "body" and left_side is not None:
                category = (
                    "leg_body_distal" if left_segment != "hip" else "leg_body_hip"
                )
            else:
                continue
            for geom_a in geometries[left]:
                for geom_b in geometries[right]:
                    pairs.append((category, geom_a, geom_b, left, right))
    if not all(
        any(pair[0] == category for pair in pairs) for category in PAIR_CATEGORIES
    ):
        raise ValueError(
            "URDF is missing expected inter-leg or leg-body collision pairs"
        )
    return model, {
        category: PairIndex(
            geometry_a=np.asarray(
                [pair[1] for pair in pairs if pair[0] == category], dtype=int
            ),
            geometry_b=np.asarray(
                [pair[2] for pair in pairs if pair[0] == category], dtype=int
            ),
            links=[(pair[3], pair[4]) for pair in pairs if pair[0] == category],
        )
        for category in PAIR_CATEGORIES
    }


def frame_distances(
    model: mujoco.MjModel,
    pairs: dict[str, PairIndex],
    joint_names: list[str],
    joint_angles: np.ndarray,
    data: mujoco.MjData | None = None,
    proximity: BulletProximity | None = None,
) -> dict[str, dict[str, object]]:
    """Return minimum signed distance for each category at one pose (meters)."""
    if len(joint_names) != len(joint_angles):
        raise ValueError("joint_names and joint_angles length differ")
    if not np.isfinite(joint_angles).all():
        raise ValueError("Nonfinite joint angle in trace")
    data = mujoco.MjData(model) if data is None else data
    for name, angle in zip(joint_names, joint_angles, strict=True):
        try:
            address = model.joint(name).qposadr
        except KeyError as exc:
            raise ValueError(f"Joint absent from URDF: {name}") from exc
        data.qpos[address] = angle
    mujoco.mj_forward(model, data)
    owned_proximity = proximity is None
    proximity = BulletProximity(model) if proximity is None else proximity
    results = {
        category: {"distance_m": math.inf, "links": None}
        for category in PAIR_CATEGORIES
    }
    try:
        proximity.update(data)
        for category, group in pairs.items():
            lower_bounds = (
                np.linalg.norm(
                    data.geom_xpos[group.geometry_a] - data.geom_xpos[group.geometry_b],
                    axis=1,
                )
                - model.geom_rbound[group.geometry_a]
                - model.geom_rbound[group.geometry_b]
            )
            for index in np.argsort(lower_bounds):
                if lower_bounds[index] >= results[category]["distance_m"]:
                    break
                distance = proximity.distance(
                    int(group.geometry_a[index]), int(group.geometry_b[index])
                )
                if distance < lower_bounds[index] - 1e-5:
                    raise ValueError(
                        "Collision distance is inconsistent with its bounding-sphere lower bound"
                    )
                if distance < results[category]["distance_m"]:
                    results[category] = {
                        "distance_m": distance,
                        "links": list(group.links[index]),
                    }
    finally:
        if owned_proximity:
            proximity.close()
    results["leg_body"] = min(
        (results["leg_body_hip"], results["leg_body_distal"]),
        key=lambda value: value["distance_m"],
    )
    del results["leg_body_hip"]
    return results


def _case_key(case: dict) -> str:
    keys = (
        "terrain",
        "level",
        "mode",
        "direction",
        "velocity",
        "teacher",
        "suite",
        "landing_mode",
        "terrain_atlas_column",
        "terrain_origin",
    )
    return json.dumps({key: case.get(key) for key in keys}, sort_keys=True)


def analyze(
    manifest_path: Path, trace_path: Path, urdf_path: Path, stride: int = 1
) -> dict:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not manifest.get("zero_update_evaluation")
        or manifest.get("optimizer_steps") != 0
        or manifest.get("training_updates") != 0
    ):
        raise ValueError("Assessment must attest zero optimizer and training updates")
    if not manifest.get("initial_state_matched"):
        raise ValueError("Assessment must attest matched initial states")
    actual_hash = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    if (
        actual_hash != EXPECTED_URDF_SHA256
        or manifest.get("urdf_sha256") != actual_hash
    ):
        raise ValueError(
            "URDF hash does not match pinned Thunder asset and assessment manifest"
        )
    model, pairs = build_model(urdf_path)
    joint_names = manifest["joint_names"]
    history = manifest["history"]
    cases = manifest["cases"]
    with np.load(trace_path, allow_pickle=False) as traces:
        angles = traces["joint_pos"]
        initial_q = traces["initial_q"]
        initial_root = traces["initial_root"]
    if (
        angles.ndim != 3
        or angles.shape[1] != len(cases)
        or angles.shape[2] != len(joint_names)
    ):
        raise ValueError("joint_pos shape does not match case and joint manifest")
    if initial_q.shape != (len(cases), len(joint_names)) or initial_root.shape != (
        len(cases),
        13,
    ):
        raise ValueError(
            "Initial joint or root state shape does not match case manifest"
        )
    if not np.isfinite(initial_q).all() or not np.isfinite(initial_root).all():
        raise ValueError("Nonfinite initial state in traces")
    if not np.allclose(initial_q, manifest["default_q"], atol=1e-6, rtol=0):
        raise ValueError(
            "Trace initial joint state differs from assessment default pose"
        )
    data = mujoco.MjData(model)
    proximity = BulletProximity(model)
    outputs: list[dict] = []
    try:
        for index, case in enumerate(cases):
            if case["id"] != index:
                raise ValueError("Case id does not index joint_pos case axis")
            last = case["end_step"]
            if not isinstance(last, int) or not 0 <= last < angles.shape[0]:
                raise ValueError(f"Invalid end_step for case {index}")
            frame_indices = list(range(0, last + 1, stride))
            if frame_indices[-1] != last:
                frame_indices.append(last)
            distances = {category: [] for category in CATEGORIES}
            minimum = {
                category: {"distance_m": math.inf, "links": None, "step": None}
                for category in CATEGORIES
            }
            for step in frame_indices:
                values = frame_distances(
                    model, pairs, joint_names, angles[step, index], data, proximity
                )
                for category, value in values.items():
                    distance = value["distance_m"]
                    distances[category].append(distance)
                    if distance < minimum[category]["distance_m"]:
                        minimum[category] = {**value, "step": step}
            output = {
                "case_id": index,
                "policy": case["policy"],
                "policy_label": history[case["policy"]]["label"],
                "teacher": case["teacher"],
                "case_key": _case_key(case),
                "terrain": case["terrain"],
                "level": case["level"],
                "end_step": last,
                "sampled_frames": len(frame_indices),
                "categories": {},
            }
            for category, values in distances.items():
                series = np.asarray(values)
                output["categories"][category] = {
                    "minimum": minimum[category],
                    "p01_distance_m": float(np.percentile(series, 1)),
                    "fraction_negative_signed_distance": float(np.mean(series < 0)),
                    "fraction_below_10mm": float(np.mean(series < 0.01)),
                }
            outputs.append(output)
    finally:
        proximity.close()
    by_key: dict[str, list[dict]] = {}
    for output in outputs:
        by_key.setdefault(output["case_key"], []).append(output)
    paired = []
    for key, group in by_key.items():
        source = [item for item in group if item["policy_label"].startswith("source")]
        candidate = [
            item for item in group if item["policy_label"].startswith("candidate")
        ]
        if len(source) != 1 or len(candidate) != 1:
            continue
        source_id, candidate_id = source[0]["case_id"], candidate[0]["case_id"]
        if not np.allclose(
            initial_q[source_id], initial_q[candidate_id], atol=1e-6, rtol=0
        ):
            raise ValueError(
                f"Unmatched initial joints for paired cases {source_id} and {candidate_id}"
            )
        if not np.allclose(
            initial_root[source_id], initial_root[candidate_id], atol=2e-4, rtol=0
        ):
            raise ValueError(
                f"Unmatched initial roots for paired cases {source_id} and {candidate_id}"
            )
        paired.append(
            {
                "case_key": key,
                "source_case_id": source_id,
                "candidate_case_id": candidate_id,
                "candidate_minus_source_minimum_m": {
                    category: candidate[0]["categories"][category]["minimum"][
                        "distance_m"
                    ]
                    - source[0]["categories"][category]["minimum"]["distance_m"]
                    for category in CATEGORIES
                },
            }
        )
    return {
        "method": "URDF collision primitives; MuJoCo FK + Bullet signed primitive distance at recorded joint positions",
        "distance_unit": "m",
        "positive_distance_meaning": "separated URDF collision envelopes",
        "negative_distance_meaning": "overlapping URDF collision envelopes; not a measured contact or force",
        "physx_self_contacts_measured": False,
        "urdf_sha256": actual_hash,
        "source_manifest": str(manifest_path),
        "source_traces": str(trace_path),
        "stride": stride,
        "cases": outputs,
        "paired": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    result = analyze(args.manifest, args.traces, args.urdf, args.stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
