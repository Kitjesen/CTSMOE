"""Render measured V6 first episodes, with FK validation and no second simulation."""
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

official = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(official))
from render_student_traces import check_fk, load_font, normalized_text, replay_pose  # noqa: E402

folder = official / 'recordings/v6_43202_20260921'
m = json.loads((folder / 'trace/manifest.json').read_text())
assert m['completed_updates'] == 43202 and len(m['cases']) == 9
with np.load(folder / 'trace/traces.npz') as f:
    d = {k: f[k] for k in ('body_pos', 'body_quat', 'joint_pos', 'joint_vel', 'velocity')}
urdf = Path('D:/inovxio/brain/train/rough/him_stairs/validation/video_2000_assets/urdf/thunder_v4.urdf')
runtime = folder / 'trace/runtime.urdf'
assert normalized_text(urdf) == normalized_text(runtime)
assert hashlib.sha256(runtime.read_bytes()).hexdigest() == m['urdf_sha256']
client = p.connect(p.DIRECT)
robot = p.loadURDF(str(urdf), flags=p.URDF_USE_INERTIA_FROM_FILE)
joints = {p.getJointInfo(robot, i)[1].decode(): i for i in range(p.getNumJoints(robot))}
links = {p.getJointInfo(robot, i)[12].decode(): i for i in range(p.getNumJoints(robot))}
joint_ids = [joints[n] for n in m['joint_names']]
checks = [(i, links[n]) for i, n in enumerate(m['body_names']) if n != 'base_link']
base = m['body_names'].index('base_link')
inertial = p.getDynamicsInfo(robot, -1)[3:5]
font, small = load_font(23), load_font(18)
maximum = 0.
terrain_body = None
active_terrain = None
visuals = {}
for name, level in {(c['terrain'], c['level']) for c in m['cases']}:
    visuals[(name, level)] = p.createVisualShape(p.GEOM_MESH,
        fileName=str(folder / f'trace/terrain_{name}_L{level}.obj'), rgbaColor=[.46, .52, .58, 1])


def frame(c, step):
    global maximum, terrain_body, active_terrain
    row = {k: d[k][step, c['id']] for k in ('body_pos', 'body_quat', 'joint_pos', 'joint_vel')}
    replay_pose(client, robot, joint_ids, row, base, inertial)
    maximum = max(maximum, check_fk(client, robot, checks, row))
    key = (c['terrain'], c['level'])
    if key != active_terrain:
        if terrain_body is not None:
            p.removeBody(terrain_body)
        terrain_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visuals[key])
        active_terrain = key
    xyz = row['body_pos'][base]
    offset = [1.3, -1.7, 1.05] if c['terrain'] == 'stairs_down' else [-.9, -1.7, 1.05]
    view = p.computeViewMatrix((xyz+offset).tolist(), (xyz+[.12, 0, -.1]).tolist(), [0, 0, 1])
    projection = p.computeProjectionMatrixFOV(49, 640/352, .03, 100)
    rgba = p.getCameraImage(320, 176, viewMatrix=view, projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER, lightDirection=[-1, -2, 4], shadow=0)[2]
    return Image.fromarray(np.asarray(rgba, dtype=np.uint8)[..., :3]).resize((640, 352), Image.Resampling.BICUBIC)


groups = [('flat', '平地起步与行进', [c for c in m['cases'] if c['terrain'] == 'flat']),
          ('stairs', '上下楼梯', [c for c in m['cases'] if c['terrain'].startswith('stairs')]),
          ('pit_failure', '1.2米高台：保留失败过程', [c for c in m['cases'] if c['terrain'] == 'pit'])]
directions = dict(forward='前进0.6 m/s', reverse='后退0.6 m/s', left='左移0.4 m/s', forward_turn='前进0.4 m/s＋转向0.4 rad/s')
videos = []
for name, title, cases in groups:
    end = max(c['end_step'] for c in cases)
    total = (end+2)//2 + 25
    width, height = (1280, 1008) if len(cases) > 1 else (640, 576)
    path = folder / (name + '.mp4')
    writer = imageio.get_writer(path, fps=25, codec='libx264', quality=8, macro_block_size=16)
    try:
        for number in range(total):
            canvas = Image.new('RGB', (width, height), (17, 23, 32))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), 'V6 学生策略｜43,202轮｜' + title, font=font, fill='white')
            draw.text((12, 42), 'Isaac实测轨迹原速回放；无噪声/推扰，固定10ms延迟', font=small, fill='#c3d2e1')
            for panel, c in enumerate(cases):
                step = min(number*2+1, c['end_step'])
                x, y = panel % 2 * 640, 78 + panel // 2 * 432
                label = directions[c['direction']] if c['terrain'] == 'flat' else (
                    ('上楼' if c['terrain'] == 'stairs_up' else '下楼') + f" L{c['level']}，台阶{c['riser_m']*100:.1f}cm，0.6 m/s"
                    if c['terrain'].startswith('stairs') else '单级1.2米，前进0.6 m/s')
                draw.text((x+10, y), label, font=font, fill='white')
                canvas.paste(frame(c, step), (x, y+32))
                vx = d['velocity'][step, c['id'], 0]
                draw.text((x+10, y+387), f"t={(step+1)*.02:.2f}s   实际前向速度 {vx:.2f}m/s", font=small, fill='#9bdcfa')
                if step == c['end_step']:
                    status = '物理失败：'+','.join(c['terms']) if c['physical'] else (
                        '完成6秒测试' if c['terrain']=='flat' else '通过4米' if c['passed_4m'] else '未通过')
                    draw.rectangle((x+5, y+34, x+635, y+63), fill='#913028' if c['physical'] else '#186b41')
                    draw.text((x+10, y+37), status+'（末帧保持）', font=small, fill='white')
            draw.text((12, height-46), '默认站姿开始；无外部航向纠偏；首回合保留，未挑选成功片段', font=small, fill='white')
            draw.text((12, height-23), '仅回放实测姿态，不运行第二套物理。不同窗口结束后保持末帧。', font=small, fill='#b4c3d2')
            writer.append_data(np.asarray(canvas))
            if number in (0, total//2, total-1):
                canvas.save(folder / f'{name}_{number:04d}.jpg')
            if number % 100 == 0:
                print('FRAME', name, number, total, flush=True)
    finally:
        writer.close()
    videos.append(dict(file=path.name, frames=total, fps=25, duration_s=total/25,
                       cases=[c['id'] for c in cases], sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
p.disconnect()
(folder / 'render_manifest.json').write_text(json.dumps(dict(videos=videos,
    max_link_error_m=maximum, physics_steps=0, checkpoint=m['checkpoint']), indent=2))
print('DONE', flush=True)
