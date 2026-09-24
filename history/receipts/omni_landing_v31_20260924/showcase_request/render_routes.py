"""CPU visual replay of actual Isaac poses, with FK agreement checked."""

import json
import sys
from pathlib import Path
import numpy as np
import pybullet as p
import imageio.v2 as imageio
from PIL import Image, ImageDraw

folder = Path(sys.argv[1])
urdf = Path(sys.argv[2])
meta = json.loads((folder / "FINALIZED.json").read_text())
with np.load(folder / "traces.npz") as archive:
    # Materialize once: NpzFile indexing otherwise decompresses full trajectories every frame.
    tr = {key: archive[key] for key in archive.files}
selected_group = int(sys.argv[3]) if len(sys.argv) > 3 else None
out = folder / ("videos_final" if selected_group is None else f"videos_final_group_{selected_group}")
out.mkdir(exist_ok=False)
client = p.connect(p.DIRECT)
robot = p.loadURDF(str(urdf), flags=p.URDF_USE_INERTIA_FROM_FILE)
jl = {p.getJointInfo(robot, i)[1].decode(): i for i in range(p.getNumJoints(robot))}
bl = {p.getJointInfo(robot, i)[12].decode(): i for i in range(p.getNumJoints(robot))}
jids = [jl[n] for n in meta["joint_names"]]
base = meta["body_names"].index("base_link")
checks = [
    (i, bl[n]) for i, n in enumerate(meta["body_names"]) if n.endswith("foot_link")
]
if not checks:
    checks = [(i, bl[n]) for i, n in enumerate(meta["body_names"]) if n != "base_link"]
dyn = p.getDynamicsInfo(robot, -1)
projection = p.computeProjectionMatrixFOV(48, 1.6, 0.03, 40)
max_fk = 0.0
summary = []
starts = range(0, len(meta["cases"]), 4) if selected_group is None else [selected_group]
for start in starts:
    clips = []
    for n in range(start, start + 4):
        case = meta["cases"][n]
        visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(folder / f"terrain_{n}.obj"),
            rgbaColor=[0.45, 0.53, 0.59, 1],
        )
        ground = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual)
        frames = []
        end = meta["ends"][n]
        last = end["step"] if end else 1249
        terminal_image = None
        for frame in range(0, 1250, 2):
            if terminal_image is not None:
                frames.append(terminal_image)
                continue
            t = min(frame, last)
            pos = tr["body_pos"][t, n, base]
            w, x, y, z = tr["body_quat"][t, n, base]
            cp, cq = p.multiplyTransforms(pos, [x, y, z, w], dyn[3], dyn[4])
            p.resetBasePositionAndOrientation(robot, cp, cq)
            for j, q in zip(jids, tr["q"][t, n]):
                p.resetJointState(robot, j, float(q))
            err = max(
                np.linalg.norm(
                    np.array(p.getLinkState(robot, j, computeForwardKinematics=True)[4])
                    - tr["body_pos"][t, n, b]
                )
                for b, j in checks
            )
            max_fk = max(max_fk, float(err))
            assert err < 0.003, (n, frame, err)
            view = p.computeViewMatrixFromYawPitchRoll(
                pos.tolist(), 2.15, 45, -24, 0, 2
            )
            rgba = p.getCameraImage(
                480,
                300,
                view,
                projection,
                renderer=p.ER_TINY_RENDERER,
                shadow=0,
                lightDirection=[-1, -2, 3],
            )[2]
            im = Image.fromarray(np.asarray(rgba, dtype=np.uint8)[:, :, :3])
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 480, 40), fill=(18, 25, 32))
            motion = "SIDE vy=0.25" if case["side"] else "FORWARD vx=0.4"
            d.text((8, 5), f"{case['name']} L{case['level']} | {motion}", fill="white")
            d.text(
                (8, 21),
                f"V31 {meta['completed']} | t={(t + 1) * 0.02:.2f}s | vx/vy={tr['velocity'][t, n, 0]:.2f}/{tr['velocity'][t, n, 1]:.2f}",
                fill="white",
            )
            if end and frame >= last:
                d.rectangle((0, 270, 480, 300), fill=(90, 30, 20))
                d.text(
                    (8, 278),
                    "END: " + ",".join(end["terms"]) + " (held frame)",
                    fill="white",
                )
            rendered = np.asarray(im)
            frames.append(rendered)
            if end and frame >= last:
                terminal_image = rendered
        clips.append(frames)
        p.removeBody(ground)
        summary.append(
            dict(
                case=case,
                end=end,
                max_hip_abs=float(np.abs(tr["q"][: last + 1, n, :4]).max()),
            )
        )
    name = meta["cases"][start]["name"]
    with imageio.get_writer(
        out / f"{name}.mp4", fps=25, codec="libx264", quality=7, macro_block_size=1
    ) as writer:
        for t in range(625):
            writer.append_data(
                np.concatenate(
                    (
                        np.concatenate((clips[0][t], clips[1][t]), axis=1),
                        np.concatenate((clips[2][t], clips[3][t]), axis=1),
                    ),
                    axis=0,
                )
            )
    Image.fromarray(
        np.concatenate(
            (
                np.concatenate((clips[0][300], clips[1][300]), axis=1),
                np.concatenate((clips[2][300], clips[3][300]), axis=1),
            ),
            axis=0,
        )
    ).save(out / f"{name}.jpg")
    print("VIDEO", name, flush=True)
(out / "RENDER_VERIFIED.json").write_text(
    json.dumps(
        dict(
            max_fk_error_m=max_fk,
            physics_steps=0,
            frames_per_video=625,
            fps=25,
            cases=summary,
        ),
        indent=2,
    )
)
p.disconnect()
