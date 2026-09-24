"""Validate replay assets and package the requested seven staircase videos."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import imageio.v2 as imageio
import numpy as np
import pybullet as p

HERE = Path(__file__).parent
ROOT = HERE / "recording_v2"
URDF = Path(
    "D:/inovxio/brain/train/rough/cts_moe_scratch/thunder/source/doso_train/doso_train/data/Robots/DOSO/thunder/urdf/thunder_v4.urdf"
)
assert (
    hashlib.sha256(URDF.read_bytes()).hexdigest()
    == "83ba02ca52863238ce4050fff7c122fb159e381fb106ba31e39e44bfc0b66af3"
)
FILES = [
    ("single_up", "单级上台阶", ROOT / "videos_final"),
    ("short_up", "短踏面上楼", ROOT / "videos_final_group_4"),
    ("long_up", "长踏面上楼", ROOT / "videos_final_group_8"),
    ("ramp_up", "坡接楼梯", ROOT / "videos_final_group_12"),
    ("short_down", "短踏面下楼", ROOT / "videos_final_group_16"),
    ("reference_up", "原参考上楼梯", ROOT / "videos_final_group_20"),
    ("reference_down", "原参考下楼梯", ROOT / "videos_final_group_24"),
]
assert all((folder / f"{name}.mp4").exists() for name, _, folder in FILES)
assert all((folder / "RENDER_VERIFIED.json").exists() for _, _, folder in FILES[1:])
meta = json.loads((ROOT / "FINALIZED.json").read_text())
with np.load(ROOT / "traces.npz") as archive:
    tr = {k: archive[k] for k in ("q", "body_pos", "body_quat")}
client = p.connect(p.DIRECT)
robot = p.loadURDF(str(URDF), flags=p.URDF_USE_INERTIA_FROM_FILE)
jl = {p.getJointInfo(robot, i)[1].decode(): i for i in range(p.getNumJoints(robot))}
bl = {p.getJointInfo(robot, i)[12].decode(): i for i in range(p.getNumJoints(robot))}
base = meta["body_names"].index("base_link")
dyn = p.getDynamicsInfo(robot, -1)
maximum = 0.0
for n in range(28):
    end = (meta["ends"][n] or {}).get("step", 1249)
    for t in sorted(set([0, end, *range(0, end, 50)])):
        pos = tr["body_pos"][t, n, base]
        w, x, y, z = tr["body_quat"][t, n, base]
        cp, cq = p.multiplyTransforms(pos, [x, y, z, w], dyn[3], dyn[4])
        p.resetBasePositionAndOrientation(robot, cp, cq)
        for name, q in zip(meta["joint_names"], tr["q"][t, n]):
            p.resetJointState(robot, jl[name], float(q))
        for b, name in enumerate(meta["body_names"]):
            if name == "base_link":
                continue
            xyz = np.array(
                p.getLinkState(robot, bl[name], computeForwardKinematics=True)[4]
            )
            maximum = max(maximum, float(np.linalg.norm(xyz - tr["body_pos"][t, n, b])))
assert maximum < 0.003, maximum
p.disconnect()
delivery = HERE / "delivery"
delivery.mkdir(exist_ok=False)
video_rows = []
for name, title, folder in FILES:
    source = folder / f"{name}.mp4"
    with imageio.get_reader(source) as reader:
        data = reader.get_meta_data()
        frames = reader.count_frames()
    assert frames == 625 and abs(data["fps"] - 25) < 0.01
    shutil.copy2(source, delivery / source.name)
    shutil.copy2(folder / f"{name}.jpg", delivery / f"{name}.jpg")
    video_rows.append(
        dict(
            name=name,
            title=title,
            frames=frames,
            fps=data["fps"],
            size=data["size"],
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    )
receipt = dict(
    checkpoint_completed=38744,
    model_unchanged=True,
    optimizer_updates=0,
    independent_sampled_all_link_fk_max_error_m=maximum,
    video=video_rows,
    note="Fixed first episodes, student mean action, training observation noise and random parameters retained. No physical simulation in renderer.",
)
(delivery / "VERIFIED.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
)
shutil.copy2(ROOT / "CASE_METRICS.json", delivery / "CASE_METRICS.json")
cards = "".join(
    f'<section><h2>{title}</h2><video controls preload="metadata" poster="{name}.jpg" src="{name}.mp4"></video><p><a href="{name}.mp4" download>下载此视频</a></p></section>'
    for name, title, _ in FILES
)
html = (
    """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V31 楼梯实测 · 38,744轮</title><style>body{background:#101820;color:#eaf0f4;font:16px/1.8 system-ui,"Microsoft YaHei";max-width:1100px;margin:32px auto;padding:0 20px}h1{font-size:30px}section{padding:20px 0;border-top:1px solid #35424e}video{width:100%;background:#000}a{color:#83d2ff}.note{background:#202e3a;padding:20px;border-radius:8px}</style><h1>V31 楼梯实测 · 38,744轮学生策略</h1><div class="note">每段25秒：左列直行0.4 m/s，右列纯侧移0.25 m/s；上排L3，下排L6。开始2秒零命令。新增路线两档台阶为10/15厘米；原参考上楼为12.5/18厘米，下楼为13/18厘米。<br>使用平均动作，保留训练观测噪声和随机参数。固定单例，不是随机成功率。END之后为末帧保持；stair_route_exit可以是到达平台或越界，不能一概视为失败。<br>轨迹来自Isaac真实仿真；视频仅回放记录姿态，无二次物理仿真。</div>"""
    + cards
    + "</html>"
)
(delivery / "index.html").write_text(html, encoding="utf-8")
with zipfile.ZipFile(
    HERE / "V31_38744_stairs_videos.zip", "x", compression=zipfile.ZIP_STORED
) as z:
    for f in delivery.iterdir():
        z.write(f, f.name)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
