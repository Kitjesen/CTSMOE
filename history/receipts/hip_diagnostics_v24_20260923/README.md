# Thunder 髋内夹时程诊断

`analyze_hips.py`只读既有评估轨迹；不加载模型、不更新优化器、不改训练代码。四髋统一换成**正值为内收**，分别报告策略原始请求、延迟/限幅后的执行目标和物理实际角的时间分布、超过软限的时长及连续时长。超过默认站姿角仅是姿态偏离，**不是故障或硬限违规**。

另报每步四髋的 `sum(max(内收请求−0.369, 0))` 的均值、p95和峰值，单位为rad/控制步。它可用于校准未来的**请求越限代价**；不等于实际关节运动，现有报告没有把它当训练奖励。实际角进入硬限0.41 rad前0.01 rad区域的时长另列，属于接近限位的风险指标。

支持V21的前缀式`SELECTED_TRACES.npz`及其`SELECTED_CASES.json`，也支持评估器的`traces.npz`/`manifest.json`批量格式。示例（从本目录执行）：

```powershell
python analyze_hips.py --trace ../stair_coverage_v21_20260922/stair_release_validation/review/SELECTED_TRACES.npz --manifest ../stair_coverage_v21_20260922/stair_release_validation/review/SELECTED_CASES.json --output V21_SELECTED_HIPS.json
```

报告的轨迹为**50 Hz控制步**。`torque_peak`等若出现，是每个控制步内已汇总的200 Hz峰值，不表示请求或角度也有200 Hz时程。`stair_platform_hull`能区分全身上平台之前和之后；它不能把接近与每一级跨阶细分。`calf_contact_peak`记录小腿地形接触，`wheel_ground_force_z`记录轮支撑；两者都不是腿间自碰撞证据。现有V21精选轨迹没有小腿接触力、自碰撞力或最小腿间距，会在报告中如实标记。请求与目标同一帧同时靠近限位仅说明饱和共现；目标可能经过延迟和变化率限制，不能以同帧共现推断精确因果。

`limited_target_minus_actual_inward`保留正负号：正值表示目标比实际更内收。`torque_peak`是**绝对值峰值**，没有扭矩方向，不能凭它诊断PD控制符号或接触力矩来源。若轨迹含`joint_vel`，另报有符号的内收方向关节速度。

源与清单SHA写入输出。固定案例结果不能当成随机成功率，更不能证明用户提到的卡腿是实机驱动故障。V23新默认角需要其**自己的**轨迹；旧V21轨迹只用于定位旧策略问题，不能冒充V23改进结果。
