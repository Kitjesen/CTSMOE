"""Render recorded Isaac student rollouts as PyBullet pose replays.

PyBullet is used only for URDF forward kinematics and TinyRenderer.  This
script never calls ``stepSimulation`` and therefore cannot change the recorded
trajectory or hide a policy failure behind a second physics simulation.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p


RENDER_SIZE = (600, 400)
OUTPUT_SIZE = (960, 640)
FPS = 25
TERMINAL_FREEZE_FRAMES = 25


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


@lru_cache(maxsize=None)
def load_font(size: int):
    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def create_visual_box(client: int, center, half_extents, color) -> int:
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=center,
        physicsClientId=client,
    )


def create_scale_ruler(client: int, base_position) -> list[int]:
    """Add visual-only 0.5 m ticks beside the start pose."""
    x, y, z = (float(value) for value in base_position)
    ground_z = z - 0.48
    bodies = [
        create_visual_box(client, [x + 1.0, y - 0.9, ground_z + 0.012], [1.0, 0.012, 0.012], [0.9, 0.2, 0.15, 1]),
        create_visual_box(client, [x, y + 0.1, ground_z + 0.014], [0.012, 1.0, 0.014], [0.15, 0.4, 0.95, 1]),
    ]
    for tick in range(5):
        tick_x = x + 0.5 * tick
        bodies.append(
            create_visual_box(
                client,
                [tick_x, y - 0.9, ground_z + 0.025],
                [0.008, 0.09, 0.025],
                [0.98, 0.9, 0.15, 1],
            )
        )
    return bodies


def terrain_label(terrain: dict) -> str:
    name = terrain["name"]
    level = terrain["level"]
    difficulty = terrain.get("difficulty")
    label = f"terrain={name}  level=L{level}"
    if difficulty is not None:
        label += f"  difficulty={float(difficulty):.4f}"
    if "actual_step_height_m" in terrain:
        label += f"  mesh riser={float(terrain['actual_step_height_m']) * 100:.1f}cm  tread~30cm"
    elif "nominal_step_height_m" in terrain:
        label += f"  nominal riser~{float(terrain['nominal_step_height_m']) * 100:.1f}cm  tread={float(terrain['tread_m']) * 100:.0f}cm"
    elif "angle_deg" in terrain:
        label += f"  slope~{float(terrain['angle_deg']):.1f}deg"
    return label


def draw_overlay(image: Image.Image, clip: dict, row: dict, *, frozen: bool, completed_updates: int) -> None:
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    small_font = load_font(16)
    width, height = image.size
    draw.rectangle((0, 0, width, 106), fill=(17, 23, 32))
    draw.text(
        (12, 6),
        f"Thunder CTS-MoE STUDENT {completed_updates} | {clip['name']} | t={float(row['time']):.2f}s | 1x",
        font=font,
        fill="white",
    )
    command = row["command"]
    velocity = row["velocity"]
    draw.text(
        (12, 33),
        f"command vx/vy/wz = {command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}",
        font=small_font,
        fill="white",
    )
    draw.text(
        (12, 57),
        f"actual  vx/vy/wz = {velocity[0]:+.2f}, {velocity[1]:+.2f} m/s, {velocity[2]:+.2f} rad/s",
        font=small_font,
        fill=(120, 220, 255),
    )
    draw.text((12, 81), terrain_label(clip["terrain"]), font=small_font, fill="white")
    draw.rectangle((0, height - 51, width, height), fill=(17, 23, 32))
    draw.text(
        (12, height - 47),
        "Real Isaac rollout / PyBullet pose replay / nominal / no DR / scale ticks 0.5 m",
        font=small_font,
        fill="white",
    )
    draw.text(
        (12, height - 25),
        (f"Distance from terrain center: {float(row['horizontal_distance_from_origin']):.2f} m / 4.00 m target"
         if "horizontal_distance_from_origin" in row else
         "No renderer physics; all robot poses and joint states are recorded measurements"),
        font=small_font,
        fill=(205, 213, 224),
    )
    terminal = bool(row.get("terminated") or row.get("truncated"))
    if terminal or frozen:
        message = "PHYSICAL TERMINATION" if row.get("terminated") else "EPISODE TIME LIMIT"
        if frozen:
            message += " - FINAL RECORDED POSE"
        draw.rectangle((12, 116, 610, 148), fill=(70, 10, 10))
        draw.text((20, 121), message, font=font, fill=(255, 100, 90))


def replay_pose(client: int, robot: int, joint_ids: list[int], row: dict, base_index: int, inertial_pose) -> None:
    base_position = row["body_pos"][base_index]
    w, x, y, z = row["body_quat"][base_index]
    inertial_position, inertial_orientation = inertial_pose
    com_position, com_orientation = p.multiplyTransforms(
        base_position,
        [x, y, z, w],
        inertial_position,
        inertial_orientation,
    )
    p.resetBasePositionAndOrientation(robot, com_position, com_orientation, physicsClientId=client)
    for joint_id, position, velocity in zip(joint_ids, row["joint_pos"], row["joint_vel"], strict=True):
        p.resetJointState(
            robot,
            joint_id,
            float(position),
            targetVelocity=float(velocity),
            physicsClientId=client,
        )


def check_fk(client: int, robot: int, checks: list[tuple[int, int]], row: dict) -> float:
    maximum = 0.0
    for body_index, link_id in checks:
        calculated = p.getLinkState(robot, link_id, computeForwardKinematics=1, physicsClientId=client)[4]
        recorded = row["body_pos"][body_index]
        maximum = max(maximum, float(np.linalg.norm(np.asarray(calculated) - np.asarray(recorded))))
    assert maximum < 0.003, f"URDF FK differs from recorded Isaac link positions by {maximum:.6f} m"
    return maximum


def render_frame(client: int, base_position, projection, *, camera_pitch: float, camera_distance: float) -> Image.Image:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[float(base_position[0]), float(base_position[1]), float(base_position[2]) - 0.12],
        distance=camera_distance,
        yaw=45,
        pitch=camera_pitch,
        roll=0,
        upAxisIndex=2,
    )
    rgba = p.getCameraImage(
        *RENDER_SIZE,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=[-1, -2, 3],
        shadow=1,
        physicsClientId=client,
    )[2]
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8)[..., :3])
    return image.resize(OUTPUT_SIZE, Image.Resampling.BICUBIC)


def render(folder: Path, urdf: Path, runtime_urdf: Path, output: Path | None = None, clip_name: str | None = None,
           *, camera_pitch: float = -20, camera_distance: float = 2.15) -> None:
    manifest_path = folder / "manifest.json"
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = source["clips"]
    if clip_name is not None:
        clips = [clip for clip in clips if clip["name"] == clip_name]
        assert len(clips) == 1, f"Unknown or duplicate clip: {clip_name}"
    assert source.get("trace_only") is True, "Expected trace_only Isaac recording"
    assert digest(runtime_urdf) == source["urdf_sha256"], "Runtime URDF hash does not match recording manifest"
    assert normalized_text(urdf) == normalized_text(runtime_urdf), "Local URDF differs beyond newline encoding"
    completed_updates = source["completed_updates"]
    assert isinstance(completed_updates, int) and completed_updates > 0
    assert source["observation_noise"] is False
    assert source["push_and_parameter_randomization"] is False

    output = folder / "pose_replay" if output is None else output
    output.mkdir(exist_ok=False)
    client = p.connect(p.DIRECT)
    assert client >= 0
    p.setAdditionalSearchPath(str(urdf.parent.resolve()), physicsClientId=client)
    robot = p.loadURDF(
        str(urdf.resolve()),
        useFixedBase=False,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client,
    )
    joint_lookup = {
        p.getJointInfo(robot, index, physicsClientId=client)[1].decode(): index
        for index in range(p.getNumJoints(robot, physicsClientId=client))
    }
    link_lookup = {
        p.getJointInfo(robot, index, physicsClientId=client)[12].decode(): index
        for index in range(p.getNumJoints(robot, physicsClientId=client))
    }
    missing_joints = sorted(set(source["joint_names"]) - joint_lookup.keys())
    missing_bodies = sorted(set(source["body_names"]) - link_lookup.keys() - {"base_link"})
    assert not missing_joints, f"Joints absent from URDF: {missing_joints}"
    assert not missing_bodies, f"Bodies absent from URDF: {missing_bodies}"
    joint_ids = [joint_lookup[name] for name in source["joint_names"]]
    checks = [(index, link_lookup[name]) for index, name in enumerate(source["body_names"]) if name != "base_link"]
    assert len(checks) == len(source["body_names"]) - 1
    base_index = source["body_names"].index("base_link")
    dynamics = p.getDynamicsInfo(robot, -1, physicsClientId=client)
    inertial_pose = (dynamics[3], dynamics[4])
    projection = p.computeProjectionMatrixFOV(48, RENDER_SIZE[0] / RENDER_SIZE[1], 0.03, 200)

    result = {
        "method": "Actual Isaac rollout; PyBullet TinyRenderer pose replay only",
        "physics_steps_in_renderer": 0,
        "source_manifest_sha256": digest(manifest_path),
        "policy_sha256": source["policy_sha256"],
        "completed_updates": source["completed_updates"],
        "runtime_urdf_sha256": digest(runtime_urdf),
        "local_urdf_sha256": digest(urdf),
        "urdf_text_equal_after_newline_normalization": True,
        "fps": FPS,
        "stride": 2,
        "record_selection": "records[1::2], plus the final record when needed",
        "playback_speed": 1,
        "render_resolution": list(RENDER_SIZE),
        "output_resolution": list(OUTPUT_SIZE),
        "terrain_mesh": "original exported OBJ loaded as visual-only geometry; no simplification",
        "camera": {"mode": "fixed world yaw/pitch; position follows recorded base position",
                   "yaw_deg": 45, "pitch_deg": camera_pitch, "distance_m": camera_distance},
        "clips": [],
        "max_link_error_m": 0.0,
    }
    combined_path = output / "combined.mp4"
    combined = imageio.get_writer(
        combined_path,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        output_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for clip in clips:
            trace_path = folder / f"{clip['name']}.json"
            records = json.loads(trace_path.read_text(encoding="utf-8"))
            assert records, f"Empty trace: {trace_path}"
            assert all(len(row["joint_pos"]) == len(joint_ids) for row in records)
            assert all(len(row["joint_vel"]) == len(joint_ids) for row in records)
            assert all(len(row["body_pos"]) == len(source["body_names"]) for row in records)
            terrain = clip["terrain"]
            terrain_path = folder / clip.get("terrain_obj", f"terrain_{terrain['name']}_L{terrain['level']}.obj")
            assert terrain_path.exists(), f"Missing original terrain mesh: {terrain_path}"
            terrain_visual = p.createVisualShape(
                p.GEOM_MESH,
                fileName=str(terrain_path.resolve()),
                meshScale=[1, 1, 1],
                rgbaColor=[0.43, 0.47, 0.52, 1],
                physicsClientId=client,
            )
            terrain_body = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=terrain_visual,
                basePosition=[0, 0, 0],
                physicsClientId=client,
            )
            ruler_bodies = create_scale_ruler(client, records[0]["body_pos"][base_index])
            selected_indices = list(range(1, len(records), 2))
            terminal = bool(records[-1].get("terminated") or records[-1].get("truncated"))
            final_supplemented = not selected_indices or selected_indices[-1] != len(records) - 1
            terminal_supplemented = terminal and final_supplemented
            if final_supplemented:
                selected_indices.append(len(records) - 1)
            assert selected_indices, f"Trace has too few records for records[1::2]: {trace_path}"

            video_path = output / f"{clip['name']}.mp4"
            writer = imageio.get_writer(
                video_path,
                fps=FPS,
                codec="libx264",
                quality=8,
                macro_block_size=16,
                output_params=["-pix_fmt", "yuv420p"],
            )
            max_error = 0.0
            last_frame = None
            key_indices = {selected_indices[0], selected_indices[len(selected_indices) // 2], selected_indices[-1]}
            try:
                for frame_number, record_index in enumerate(selected_indices):
                    row = records[record_index]
                    replay_pose(client, robot, joint_ids, row, base_index, inertial_pose)
                    max_error = max(max_error, check_fk(client, robot, checks, row))
                    frame = render_frame(client, row["body_pos"][base_index], projection,
                                         camera_pitch=camera_pitch, camera_distance=camera_distance)
                    draw_overlay(frame, clip, row, frozen=False, completed_updates=completed_updates)
                    frame_array = np.asarray(frame)
                    writer.append_data(frame_array)
                    combined.append_data(frame_array)
                    last_frame = frame
                    if record_index in key_indices:
                        frame.save(output / f"{clip['name']}_record_{record_index:04d}.jpg", quality=92)
                    if frame_number % 100 == 0:
                        print("FRAME", clip["name"], frame_number, flush=True)
                if terminal:
                    frozen_frame = last_frame.copy()
                    draw_overlay(frozen_frame, clip, records[-1], frozen=True, completed_updates=completed_updates)
                    frozen_array = np.asarray(frozen_frame)
                    for _ in range(TERMINAL_FREEZE_FRAMES):
                        writer.append_data(frozen_array)
                        combined.append_data(frozen_array)
                    frozen_frame.save(output / f"{clip['name']}_terminal_frozen.jpg", quality=92)
            finally:
                writer.close()
                for body in ruler_bodies:
                    p.removeBody(body, physicsClientId=client)
                p.removeBody(terrain_body, physicsClientId=client)

            result["max_link_error_m"] = max(result["max_link_error_m"], max_error)
            result["clips"].append(
                {
                    "name": clip["name"],
                    "terrain": terrain,
                    "terrain_obj": terrain_path.name,
                    "terrain_obj_sha256": digest(terrain_path),
                    "trace_sha256": digest(trace_path),
                    "source_records": len(records),
                    "rendered_motion_frames": len(selected_indices),
                    "terminal": terminal,
                    "terminal_record_supplemented": terminal_supplemented,
                    "final_record_supplemented": final_supplemented,
                    "terminal_freeze_frames": TERMINAL_FREEZE_FRAMES if terminal else 0,
                    "max_link_error_m": max_error,
                    "video": video_path.name,
                    "video_sha256": digest(video_path),
                }
            )
            print("RENDER_COMPLETE", clip["name"], "FK_ERROR_M", f"{max_error:.9f}", flush=True)
    finally:
        combined.close()
        p.disconnect(client)

    result["combined_video"] = combined_path.name
    result["combined_video_sha256"] = digest(combined_path)
    (output / "render_manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("POSE_REPLAY_COMPLETE", output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True, help="Folder containing manifest.json and trace files")
    parser.add_argument("--urdf", type=Path, required=True, help="Local URDF used for pose replay")
    parser.add_argument("--runtime-urdf", type=Path, required=True, help="Exact URDF copied from the Isaac runtime")
    parser.add_argument("--output", type=Path, help="Output folder (default: FOLDER/pose_replay)")
    parser.add_argument("--clip", help="Render one named clip, allowing independent clips to run concurrently")
    parser.add_argument("--camera-pitch", type=float, default=-20, help="Downward camera pitch in degrees")
    parser.add_argument("--camera-distance", type=float, default=2.15, help="Camera distance in meters")
    args = parser.parse_args()
    render(
        args.folder.resolve(),
        args.urdf.resolve(),
        args.runtime_urdf.resolve(),
        args.output.resolve() if args.output is not None else None,
        args.clip,
        camera_pitch=args.camera_pitch,
        camera_distance=args.camera_distance,
    )


if __name__ == "__main__":
    main()
