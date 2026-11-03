"""Summarize the completed fixed-case assessment without changing its gates."""
import collections
import csv
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
OUT = HERE / 'requested_evaluation_33865'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    final = read('FINALIZED.json')
    assert final['optimizer_updates'] == 0 and final['hashes_unchanged']
    flat = read('paired/flat25/analysis_v14.json')['cases']
    flat_metrics = {}
    for label in ('source31831', 'source31831_teacher', 'candidate', 'candidate_teacher'):
        rows = [x for x in flat if x['label'] == label]
        flat_metrics[label] = dict(
            count=len(rows), pose=sum(x['pose_pass'] for x in rows),
            tracking=sum(x['tracking_pass'] for x in rows), joint=sum(x['pass'] for x in rows),
            physical=sum(x['physical'] for x in rows),
            first_peak_rms_mean=statistics.mean(x['first_peak_rms'] for x in rows),
            first_peak_rms_max=max(x['first_peak_rms'] for x in rows),
            end_rms_mean=statistics.mean(x['last_rms'] for x in rows),
            end_height_mean=statistics.mean(x['last_height'] for x in rows),
            leg_torque_peak=max(x['leg_torque_peak_nm'] for x in rows),
            wheel_torque_peak=max(x['wheel_torque_peak_nm'] for x in rows),
            leg_speed_peak=max(x['leg_speed_peak_rad_s'] for x in rows),
            wheel_speed_peak=max(x['wheel_speed_peak_rad_s'] for x in rows),
            failures=[dict(direction=x['direction'], mode=x['mode'], pose=x['pose_pass'],
                           xy_error=x['last_xy_error'], yaw_error=x['last_yaw_error'])
                      for x in rows if not x['pass']])
    terrain = read('paired/terrain83/summary.json')['terrains']
    braking = {}
    for label in ('source33174', 'source33174_teacher', 'candidate', 'candidate_teacher'):
        rows = read(f'focused/brake25/BRAKE_METRICS_{label}.json')['cases']
        events = [e for r in rows for e in r['stop_events']]
        eligible = [e for e in events if e['reached_command_before_stop']]
        passed = [e for e in events if e['qualified_stop']]
        braking[label] = dict(cases=len(rows), physical=sum(x['physical'] for x in rows),
                             events=len(events), entry_qualified=len(eligible),
                             qualified_stops=len(passed),
                             median_stop_s=statistics.median(e['stable_stop_time_s'] for e in passed),
                             max_stop_s=max(e['stable_stop_time_s'] for e in passed),
                             max_distance_m=max(e['path_until_stable_stop_m'] for e in passed),
                             counts_by_axis={axis: dict(
                                 total=sum(r['direction'].startswith(axis) * len(r['stop_events']) for r in rows),
                                 entry=sum(e['reached_command_before_stop'] for r in rows
                                           if r['direction'].startswith(axis) for e in r['stop_events']),
                                 stopped=sum(e['qualified_stop'] for r in rows
                                             if r['direction'].startswith(axis) for e in r['stop_events']))
                                 for axis in ('longitudinal', 'lateral', 'yaw')})
    stair_manifest = read('focused/stairend/manifest.json')
    stair_rows = read('focused/stairend/STAIR_END_METRICS.json')['cases']
    stairs = []
    for policy, history in enumerate(stair_manifest['history']):
        for name in ('stairs_up', 'stairs_down'):
            for mode in ('continue', 'stop'):
                rows = [r for r in stair_rows if r['policy'] == policy
                        and r['terrain'] == name and r['landing_mode'] == mode]
                stairs.append(dict(label=history['label'], terrain=name, mode=mode, count=len(rows),
                                   physical=sum(r['physical'] for r in rows),
                                   end_reasons=dict(collections.Counter(r['end_reason'] for r in rows)),
                                   support_seen=sum(r['supported_platform_reached'] for r in rows),
                                   positive_world_vz_peak=max(r['window_positive_vz_peak'] for r in rows)))
    with (OUT / 'paired/flat25/joint_assessment.csv').open(encoding='utf-8-sig') as f:
        joint_rows = list(csv.DictReader(f))
    joints = []
    for name in dict.fromkeys(r['joint'] for r in joint_rows):
        rows = [r for r in joint_rows if r['label'] == 'candidate' and r['joint'] == name]
        first = [r for r in rows if r['window'] == 'startup_0_1']
        full = [r for r in rows if r['window'] == 'full_episode']
        joints.append(dict(joint=name, startup_actual_peak=max(float(r['actual_peak_abs_rad']) for r in first),
                           full_actual_peak=max(float(r['actual_peak_abs_rad']) for r in full),
                           full_requested_peak=max(float(r['request_peak_abs_rad']) for r in full),
                           full_target_peak=max(float(r['target_peak_abs_rad']) for r in full)))
    lifts = [r for r in read('paired/flat25/OMNI_LIFT_METRICS.json')['cases']
             if r['label'].startswith('candidate') and r['direction'] in ('left', 'right', 'yaw_left', 'yaw_right')]
    report = dict(scope='652 fixed deterministic-protocol cases, not random success rates',
                  completed=33865, flat25=flat_metrics, terrain=terrain, braking=braking,
                  stair_terminal=stairs, student_joints=joints, lift=lifts,
                  original_verdict=read('VERDICT.json'), flat20_applied=False,
                  allocation_recommendation='Defer direct40-to20 change while low-speed lateral tracking and reverse braking remain unresolved. Preserve20 as a later controlled candidate.',
                  training_running=False)
    (OUT / 'REVIEW_METRICS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 33865轮即时评估与平地20%判断', '',
             '已完成652个固定案例：原332例九地形、80例完整平地、144例刹车、96例楼梯末段。教师和学生均测；零优化器更新，检查点哈希未变。结果不是随机成功率或实机验收。', '',
             '**判断：常规平地姿态保持改善，但新速度范围尚未完整掌握，不建议此刻直接从40%降到20%。20%候选已准备，未应用；训练保存在33865轮并停止。**', '',
             '## 完整25秒平地', '',
             '| 策略 | 姿态 | 跟踪 | 联合 | 物理终止 | 平均起步RMS峰值rad | 最差起步RMS峰值rad |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label, r in flat_metrics.items():
        lines.append(f"| {label} | {r['pose']}/20 | {r['tracking']}/20 | {r['joint']}/20 | {r['physical']} | {r['first_peak_rms_mean']:.3f} | {r['first_peak_rms_max']:.3f} |")
    lines += ['', '学生剩下两例为斜向后退跟踪失败，姿态通过。当前同批源31831联合15/20，与旧报告17/20略有不同；不能将跨批次计数差当精确学习增益。', '',
              '## 九地形', '', '| 地形 | 学生源31831 | 学生33865 | 教师源31831 | 教师33865 |', '|---|---:|---:|---:|---:|']
    labels = ('source31831', 'candidate', 'source31831_teacher', 'candidate_teacher')
    lookup = {(r['label'], r['terrain']): r for r in terrain}
    for name in ('flat', 'stairs_up', 'stairs_down', 'slope_up', 'slope_down', 'wave', 'rough_slope', 'obstacles', 'pit'):
        values = [f"{lookup[label,name]['passed']}/{lookup[label,name]['count']}" for label in labels]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '普通楼梯6/6是旧4米路线通过；不能代替末端平台恢复。高台学生7/21、教师8/21，最高稳定成功仍35厘米，45厘米及以上本批无成功。', '',
              '## 刹车及速度覆盖', '', '| 策略 | 事件数 | 停前速度合格 | 合格且稳定停住 | 物理终止案例 | 合格组停车中位数s |', '|---|---:|---:|---:|---:|---:|']
    for label, r in braking.items():
        lines.append(f"| {label} | {r['events']} | {r['entry_qualified']} | {r['qualified_stops']} | {r['physical']} | {r['median_stop_s']:.2f} |")
    lines += ['', '每条完整轨迹有两次停车事件，事件不等于独立随机试验。停前条件要求最近1秒至少80%样本及末样本达到指令的±20%，非命令轴也需稳定。未满足入口条件不能冒充高速刹车成功。低于0.1m/s平移、0.15rad/s转向持续0.5秒定义稳定停止，不是精确数学零。', '',
              '学生案例74（2m/s前进、停车、2m/s后退、停车）：第一停前实际1.864m/s，0.34秒停住；第二停前实际后退1.836m/s，15秒给全零后轮速峰值46.129rad/s，触发actuator_violation，未形成稳定停止。', '',
              '学生纯侧移24次停车入口仅5次合格。0.2m/s右移在direct/settle的首个停车前侧速约0.00066/-0.000003m/s，几乎没有执行侧移；0.2m/s左移约0.076/0.093m/s。高侧速有些已接近目标，但存在Yaw漂移或速度波动。不能把19次入口不合格全部称为刹车失败。', '',
              '左纯Yaw仍能跟踪，但direct/settle均无达到既定3厘米净空等条件的有效抬落轮事件；最大净空约2.3–2.9厘米，不等于完全不抬轮。右Yaw有效事件50/43，方向不对称需要检查。', '',
              '## 楼梯真实末段专项', '',
              '四策略共96例均未触发物理终止；使用专门封顶外平台，区别于训练及旧4米路线。学生上楼末段向上速度峰值约0.755m/s，切换前33174约0.970m/s；下楼约0.486m/s，仍存在回弹，不能宣称已消除。', '',
              '停车分支：学生上楼2/6达到2秒连续支撑，下楼3/6。上楼的另3个6级案例没有触发有效平台支撑/停车条件就到达安全边界，因此不能描述成“收到停车命令后全部停不住”，应先复核平台触发/扫描几何；继续前进分支到安全边界是预期结束，不用停车成功标准评价。', '',
              '## 12个腿关节（学生、20个常规平地案例）', '',
              '| 关节 | 起步实际偏移峰值rad | 全回合实际峰值rad | 请求峰值rad | 限幅目标峰值rad |', '|---|---:|---:|---:|---:|']
    for r in joints:
        lines.append(f"| {r['joint']} | {r['startup_actual_peak']:.3f} | {r['full_actual_peak']:.3f} | {r['full_requested_peak']:.3f} | {r['full_target_peak']:.3f} |")
    lines += ['', '各列是各自跨案例最大值，不一定发生在同一时刻。原始请求仍有较大髋偏移（左后约1.79rad），实际姿态改善不能等同于请求已规整。原始逐关节CSV同时保存对应时刻的请求、目标和实际角度。', '',
              '## 配比决定与下一步', '',
              '原有防崩门控没有触发；本次暂停是用户要求即时评估，不是伪称门控拒绝。扩展测试揭示低速侧移、后退急停及末段恢复仍有缺口，所以暂缓直接平地20%。先定位低速命令/奖励阈值与急停轮速轨迹，再准备单因素有限修复；不要通过放松执行器终止掩盖46rad/s轮速。20%只作为后续保留完整纯侧移/纯Yaw/启停覆盖的对照配额，不同时改奖励。', '',
              '总已用更新2042（含历史丢失8），剩7958；本次评估不消耗更新预算。全部原轨迹保存在远端requested_evaluation_33865，关键报告、逐关节CSV与选取时程已取回本地。']
    (OUT / 'EVALUATION_REVIEW.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'cases':652, 'flat_student_joint':flat_metrics['candidate']['joint'],
                      'student_brake_physical':braking['candidate']['physical'],
                      'flat20_applied':False}))


if __name__ == '__main__':
    main()
