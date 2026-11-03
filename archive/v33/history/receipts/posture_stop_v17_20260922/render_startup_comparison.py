"""Compare measured first episodes; visual replay only, no second physics simulation."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

OFFICIAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OFFICIAL))
from render_student_traces import check_fk, load_font, normalized_text, replay_pose  # noqa: E402


def main():
    folder = Path(sys.argv[1])
    manifest = json.loads((folder / 'forward_manifest.json').read_text(encoding='utf-8'))
    with np.load(folder / 'forward.npz') as file:
        data = dict(file)
    cases = manifest['cases']
    assert len(cases) == 2 and manifest['optimizer_steps'] == 0
    assert all(c['velocity'] == [.6, 0., 0.] and not c['teacher'] for c in cases)
    names = manifest['joint_names']
    legs = [names.index(f'{side}_{joint}_joint') for side in ('fl', 'fr', 'rl', 'rr')
            for joint in ('hip', 'thigh', 'calf')]
    assert len(legs) == 12
    defaults = np.asarray(manifest['default_q'])
    dt = manifest['dt']
    times = (np.arange(len(data['joint_pos'])) + 1) * dt
    errors = data['joint_pos'][:, :, legs] - defaults[legs]
    rms = np.sqrt(np.mean(errors ** 2, axis=-1))
    rows, summaries = [], []
    completed = [next(h['fields']['completed_updates'] for h in manifest['history']
                      if h['path'] == case['checkpoint'] and not h['teacher']) for case in cases]
    if completed != [31831, 32831]:
        raise ValueError(f'Wrong source/candidate recording: {completed}')
    labels = [f'源策略 {completed[0]:,}轮', f'V17 {completed[1]:,}轮']
    for i, case in enumerate(cases):
        end = case['end_step'] + 1
        first = min(round(1 / dt), end)
        for j, leg in enumerate(legs):
            target = data['target'][:first, i, leg]
            delta = np.diff(np.r_[data['initial_q'][i, leg], target]) / dt
            rows.append(dict(policy=labels[i], joint=names[leg],
                actual_peak_abs_rad=float(abs(errors[:first, i, j]).max()),
                target_peak_abs_rad=float(abs(target - defaults[leg]).max()),
                first_target_delta_rad=float(target[0] - data['initial_q'][i, leg]),
                target_rate_peak_rad_s=float(abs(delta).max()),
                torque_peak_nm=float(data['torque_peak'][:first, i, leg].max()),
                speed_peak_rad_s=float(data['speed_peak'][:first, i, leg].max())))
        summaries.append(dict(policy=labels[i], physical=case['physical'], seconds=case['seconds'],
            startup_rms_peak_rad=float(rms[:first, i].max()),
            startup_single_joint_peak_rad=float(abs(errors[:first, i]).max()),
            startup_height_min_m=float(data['root_clearance'][:first, i].min()),
            startup_vx_peak_m_s=float(data['velocity'][:first, i, 0].max()),
            tail_rms_mean_rad=float(rms[max(0, end - 250):end, i].mean()),
            tail_vx_mean_m_s=float(data['velocity'][max(0, end - 250):end, i, 0].mean())))
    with (folder / 'startup_per_joint.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'startup_metrics.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    colors = ['#1875b5', '#dc513d']
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for i in range(2):
        n = min(300, cases[i]['end_step'] + 1)
        for ax, values in zip(axes, (rms[:n, i], data['root_clearance'][:n, i], data['velocity'][:n, i, 0])):
            ax.plot(times[:n], values, color=colors[i], label=labels[i])
    for ax, title in zip(axes, ['12腿关节相对默认角 RMS（rad）', '机身相对地面高度（m）', '实际前向速度（m/s）']):
        ax.set_ylabel(title)
        ax.grid(alpha=.2)
        ax.legend()
    axes[1].axhline(.46, color='gray', ls='--')
    axes[2].axhline(.6, color='gray', ls='--')
    axes[-1].set_xlabel('直接给速后时间（秒）')
    fig.suptitle('相同默认站姿直接给0.6 m/s；学生首回合，无噪声/推扰，固定10 ms延迟')
    fig.tight_layout()
    fig.savefig(folder / 'startup_summary.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(16, 13), sharex=True)
    request_ids = list(manifest['request_joint_ids'])
    for j, ax in enumerate(axes.flat):
        leg = legs[j]
        for i in range(2):
            n = min(150, cases[i]['end_step'] + 1)
            ax.plot(times[:n], errors[:n, i, j], color=colors[i], label=labels[i] + ' 实际')
        n = min(150, cases[1]['end_step'] + 1)
        ax.plot(times[:n], data['target'][:n, 1, leg] - defaults[leg], '--', color='#f2992e', label='V17 限制后目标')
        req = data['requested'][:n, 1, request_ids.index(leg)] - defaults[leg]
        ax.plot(times[:n], req, ':', color='#8a58a3', alpha=.65, label='V17 原始请求')
        ax.axhline(0, color='gray', lw=.6)
        ax.set_title(['左前', '右前', '左后', '右后'][j // 3]
                     + ['髋', '大腿', '小腿'][j % 3] + ' / ' + names[leg])
        ax.grid(alpha=.2)
        ax.set_ylabel('相对默认角（rad）')
        ax.set_xlabel('秒')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('逐关节起步时程：请求、限制后目标、实际角；各图纵轴独立，未裁剪大请求')
    fig.tight_layout()
    fig.savefig(folder / 'startup_12_joints.png', dpi=145)
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
    for j, ax in enumerate(axes.flat):
        for i in range(2):
            n = min(150, cases[i]['end_step'] + 1)
            ax.plot(times[:n], errors[:n, i, j], color=colors[i], label=labels[i])
        ax.axhline(0, color='gray', ls='--', lw=.8)
        ax.set_title(['左前', '右前', '左后', '右后'][j // 3]
                     + ['髋', '大腿', '小腿'][j % 3])
        ax.set_ylabel('实际角相对默认角（rad）')
        ax.set_xlabel('秒')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle('12关节实际姿态对照：起步前3秒；零线为默认站姿，各图纵轴独立')
    fig.tight_layout()
    fig.savefig(folder / 'startup_actual_angles.png', dpi=145)
    plt.close(fig)

    urdf = Path('D:/inovxio/brain/train/rough/him_stairs/validation/video_2000_assets/urdf/thunder_v4.urdf')
    runtime = folder / 'runtime.urdf'
    assert normalized_text(urdf) == normalized_text(runtime)
    assert hashlib.sha256(runtime.read_bytes()).hexdigest() == manifest['urdf_sha256']
    client = p.connect(p.DIRECT)
    robot = p.loadURDF(str(urdf), flags=p.URDF_USE_INERTIA_FROM_FILE)
    joint_map = {p.getJointInfo(robot, j)[1].decode(): j for j in range(p.getNumJoints(robot))}
    links = {p.getJointInfo(robot, j)[12].decode(): j for j in range(p.getNumJoints(robot))}
    joint_ids = [joint_map[n] for n in names]
    checks = [(j, links[n]) for j, n in enumerate(manifest['body_names']) if n != 'base_link']
    base = manifest['body_names'].index('base_link')
    inertial = p.getDynamicsInfo(robot, -1)[3:5]
    surface = p.createVisualShape(p.GEOM_BOX, halfExtents=[40, 40, .01], rgbaColor=[.68, .72, .75, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=surface, basePosition=[0, 0, -.01])
    for x in range(-1, 22):
        line = p.createVisualShape(p.GEOM_BOX, halfExtents=[.005, 1.6, .001], rgbaColor=[.45, .49, .52, 1])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=line, basePosition=[x, 0, .001])
    font, small = load_font(23), load_font(18)
    projection = p.computeProjectionMatrixFOV(47, 640/440, .03, 100)
    maximum = 0.
    startup_frames = []
    output = folder / 'forward_comparison_25s.mp4'
    with imageio.get_writer(output, fps=25, codec='libx264', quality=8, macro_block_size=16) as writer:
        for number in range(625):
            step = number * 2
            canvas = Image.new('RGB', (1280, 640), '#111720')
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 10), '平地起步对照｜立即给前进0.6 m/s｜1倍速', font=font, fill='white')
            draw.text((18, 43), '实测Isaac轨迹回放；默认站姿，无噪声/推扰，固定10 ms延迟；不挑选回合', font=small, fill='#bfd0df')
            for i, case in enumerate(cases):
                k = min(step, case['end_step'])
                row = {key: data[key][k, i] for key in ('body_pos', 'body_quat', 'joint_pos', 'joint_vel')}
                replay_pose(client, robot, joint_ids, row, base, inertial)
                maximum = max(maximum, check_fk(client, robot, checks, row))
                xyz = row['body_pos'][base]
                view = p.computeViewMatrix((xyz + [-1., -1.8, .9]).tolist(), (xyz + [.08, 0, -.12]).tolist(), [0, 0, 1])
                # Render pixels only at half resolution; keep every measured
                # 25 Hz frame. Removing visual shadows does not change poses.
                rgba = p.getCameraImage(320, 220, viewMatrix=view, projectionMatrix=projection,
                                       renderer=p.ER_TINY_RENDERER, lightDirection=[-1, -2, 4], shadow=0)[2]
                panel = Image.fromarray(np.asarray(rgba, dtype=np.uint8)[..., :3])
                canvas.paste(panel.resize((640, 440), Image.Resampling.BICUBIC), (640 * i, 105))
                draw.text((640*i + 15, 76), labels[i], font=font, fill=colors[i])
                draw.text((640*i + 15, 553), f't={times[k]:.2f}s  vx={data["velocity"][k,i,0]:.2f}m/s  高度={data["root_clearance"][k,i]:.3f}m', font=small, fill='white')
                draw.text((640*i + 15, 580), f'关节RMS={rms[k,i]:.3f}rad  最大偏移={abs(errors[k,i]).max():.3f}rad', font=small, fill='white')
                if k == case['end_step'] and case['physical']:
                    draw.text((640*i + 15, 110), '物理失败，保持末帧', font=font, fill='#cf3020')
            draw.text((18, 611), '渲染不运行第二套物理；左右同一视角；地面参考线间隔1米', font=small, fill='#b6c8d7')
            writer.append_data(np.asarray(canvas))
            if number < 75:
                startup_frames.append(canvas.copy())
            if number in (0, 5, 12, 24, 49, 74, 124, 624):
                canvas.save(folder / f'comparison_frame_{number:04d}.jpg')
            if number % 100 == 0:
                print('RENDER', number, flush=True)
    with imageio.get_writer(folder / 'startup_comparison_slow.mp4', fps=25, codec='libx264', quality=8, macro_block_size=16) as writer:
        for canvas in startup_frames:
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 1280, 40), fill='#111720')
            draw.text((18, 10), '起步前3秒｜0.25倍慢放（标注时间为真实仿真时间）', font=font, fill='white')
            for _ in range(4):
                writer.append_data(np.asarray(canvas))
    p.disconnect()
    (folder / 'render_manifest.json').write_text(json.dumps(dict(max_fk_error_m=maximum,
        physics_steps=0, frames=625, fps=25, slow_frames=300,
        videos={n: hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in
                ('forward_comparison_25s.mp4', 'startup_comparison_slow.mp4')}), indent=2))
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
