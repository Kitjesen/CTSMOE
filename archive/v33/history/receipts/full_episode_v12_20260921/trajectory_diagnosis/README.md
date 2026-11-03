# V12 pilot300 轨迹诊断

## 口径与结论

本报告分析的是协议预先枚举的 fixed cases，不是随机重复试验，因此 `pass/count` 仅表示固定 case 覆盖，不解释为成功概率。V6 student 只出现在 flat25 参考中。

normal 共 332 个 case；flat25 共 100 个 case。原始输入哈希与逐 case 数据在 `diagnosis.json`。

## 主要发现

- 九地形总计：V12 student 62/83，V9 student 60/83；逐 case 实际为 10 个 gain、8 个 loss。V12 teacher 为 67/83，相对 V9 teacher 是 8 gain、1 loss。
- flat25：所有策略的 100/100 个 case 都完成 25s，无 physics termination。V12 student 15/20，V9 student 10/20，V6 student 参考 16/20；V12 teacher 17/20。
- V12 student 直接前进的 0–1s qRMS 峰为 0.4643rad@0.28s，20–25s 峰降为 0.1850rad；settle 前进命令后首秒峰仅 0.1289rad，并通过 gate。
- 高台：V12 student 在 0.25/0.35/0.45/0.60m 的三个固定速度均通过；0.15m 的 0.6m/s 从 V9 pass 变为 `pit_outside`，0.90m 与 1.20m 三个速度均未通过。
- 楼梯末段最大前向速度峰峰值出现在 V12_45011 stairs_up L9 1.2m/s：1.4884m/s；该 case pass=False，terms=['actuator_violation']，末段腿关节速度峰 31.5762rad/s。

## 九地形矩阵

|策略|地形|pass/count|physics|straight|
|---|---:|---:|---:|---:|
|V12_45011|flat|14/20|0|0|
|V12_45011|obstacles|5/6|1|4|
|V12_45011|pit|14/21|3|0|
|V12_45011|rough_slope|5/6|1|2|
|V12_45011|slope_down|6/6|0|4|
|V12_45011|slope_up|3/6|2|2|
|V12_45011|stairs_down|6/6|0|4|
|V12_45011|stairs_up|4/6|1|2|
|V12_45011|wave|5/6|1|4|
|V12_45011_teacher|flat|14/20|0|0|
|V12_45011_teacher|obstacles|6/6|0|4|
|V12_45011_teacher|pit|15/21|5|0|
|V12_45011_teacher|rough_slope|6/6|0|3|
|V12_45011_teacher|slope_down|6/6|0|4|
|V12_45011_teacher|slope_up|3/6|3|2|
|V12_45011_teacher|stairs_down|6/6|0|3|
|V12_45011_teacher|stairs_up|5/6|0|4|
|V12_45011_teacher|wave|6/6|0|4|
|V9_44711|flat|14/20|0|0|
|V9_44711|obstacles|6/6|0|5|
|V9_44711|pit|7/21|11|0|
|V9_44711|rough_slope|6/6|0|5|
|V9_44711|slope_down|6/6|0|6|
|V9_44711|slope_up|3/6|1|2|
|V9_44711|stairs_down|6/6|0|5|
|V9_44711|stairs_up|6/6|0|4|
|V9_44711|wave|6/6|0|4|
|V9_44711_teacher|flat|13/20|0|0|
|V9_44711_teacher|obstacles|5/6|0|4|
|V9_44711_teacher|pit|10/21|6|0|
|V9_44711_teacher|rough_slope|6/6|0|6|
|V9_44711_teacher|slope_down|6/6|0|6|
|V9_44711_teacher|slope_up|2/6|1|1|
|V9_44711_teacher|stairs_down|6/6|0|6|
|V9_44711_teacher|stairs_up|6/6|0|6|
|V9_44711_teacher|wave|6/6|0|2|

### 学生 V12 对 V9

逐 case 配对 83 个：新增通过 10，失去通过 8。
失去 case：
- flat L0 yaw_right [0.0, 0.0, -0.6]，V12 terms=[] physical=False。
- flat L0 reverse [-0.6, 0.0, 0.0]，V12 terms=[] physical=False。
- obstacles L6 forward [0.6, 0.0, 0.0]，V12 terms=['illegal_contact'] physical=True。
- pit L0 forward [0.6, 0.0, 0.0]，V12 terms=['pit_outside'] physical=False。
- rough_slope L6 forward [1.2, 0.0, 0.0]，V12 terms=['actuator_violation'] physical=True。
- stairs_up L9 forward [0.3, 0.0, 0.0]，V12 terms=['time_out'] physical=False。
- stairs_up L9 forward [1.2, 0.0, 0.0]，V12 terms=['actuator_violation'] physical=True。
- wave L9 forward [0.3, 0.0, 0.0]，V12 terms=['actuator_violation'] physical=True。
新增 case：
- flat L0 forward [0.6, 0.0, 0.0]，V12 terms=[] physical=False。
- flat L0 forward_turn [0.4, 0.0, 0.4]，V12 terms=[] physical=False。
- pit L4 forward [0.6, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L4 forward [0.9, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L6 forward [0.3, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L6 forward [0.6, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L6 forward [0.9, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L9 forward [0.3, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L9 forward [0.6, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L9 forward [0.9, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。

### 教师 V12 对 V9

逐 case 配对 83 个：新增通过 8，失去通过 1。
失去 case：
- stairs_up L9 forward [0.3, 0.0, 0.0]，V12 terms=['time_out'] physical=False。
新增 case：
- flat L0 diagonal_back [-0.4, -0.3, 0.0]，V12 terms=[] physical=False。
- obstacles L6 forward [0.3, 0.0, 0.0]，V12 terms=[] physical=False。
- pit L0 forward [0.3, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L0 forward [0.6, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L6 forward [0.9, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L9 forward [0.6, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- pit L9 forward [0.9, 0.0, 0.0]，V12 terms=['pit_success'] physical=False。
- slope_up L6 forward [0.6, 0.0, 0.0]，V12 terms=[] physical=False。

## 高台实际高度

|策略|高度 m|pass/count|physics|成功速度|失败速度|
|---|---:|---:|---:|---|---|
|V12_45011|0.15|2/3|0|[0.3, 0.9]|[0.6]|
|V12_45011|0.25|3/3|0|[0.3, 0.6, 0.9]|[]|
|V12_45011|0.35|3/3|0|[0.3, 0.6, 0.9]|[]|
|V12_45011|0.45|3/3|0|[0.3, 0.6, 0.9]|[]|
|V12_45011|0.60|3/3|0|[0.3, 0.6, 0.9]|[]|
|V12_45011|0.90|0/3|2|[]|[0.3, 0.6, 0.9]|
|V12_45011|1.20|0/3|1|[]|[0.3, 0.6, 0.9]|
|V12_45011_teacher|0.15|3/3|0|[0.3, 0.6, 0.9]|[]|
|V12_45011_teacher|0.25|3/3|0|[0.3, 0.6, 0.9]|[]|
|V12_45011_teacher|0.35|3/3|0|[0.3, 0.6, 0.9]|[]|
|V12_45011_teacher|0.45|3/3|0|[0.3, 0.6, 0.9]|[]|
|V12_45011_teacher|0.60|3/3|0|[0.3, 0.6, 0.9]|[]|
|V12_45011_teacher|0.90|0/3|3|[]|[0.3, 0.6, 0.9]|
|V12_45011_teacher|1.20|0/3|2|[]|[0.3, 0.6, 0.9]|
|V9_44711|0.15|3/3|0|[0.3, 0.6, 0.9]|[]|
|V9_44711|0.25|3/3|0|[0.3, 0.6, 0.9]|[]|
|V9_44711|0.35|1/3|0|[0.3]|[0.6, 0.9]|
|V9_44711|0.45|0/3|2|[]|[0.3, 0.6, 0.9]|
|V9_44711|0.60|0/3|3|[]|[0.3, 0.6, 0.9]|
|V9_44711|0.90|0/3|3|[]|[0.3, 0.6, 0.9]|
|V9_44711|1.20|0/3|3|[]|[0.3, 0.6, 0.9]|
|V9_44711_teacher|0.15|1/3|1|[0.9]|[0.3, 0.6]|
|V9_44711_teacher|0.25|3/3|0|[0.3, 0.6, 0.9]|[]|
|V9_44711_teacher|0.35|3/3|0|[0.3, 0.6, 0.9]|[]|
|V9_44711_teacher|0.45|2/3|0|[0.3, 0.6]|[0.9]|
|V9_44711_teacher|0.60|1/3|0|[0.3]|[0.6, 0.9]|
|V9_44711_teacher|0.90|0/3|3|[]|[0.3, 0.6, 0.9]|
|V9_44711_teacher|1.20|0/3|2|[]|[0.3, 0.6, 0.9]|

## flat25 完整性

|策略|模式|完成25s/count|pass|physics|
|---|---:|---:|---:|---:|
|V12_45011|direct|10/10|7|0|
|V12_45011|settle|10/10|8|0|
|V12_45011_teacher|direct|10/10|8|0|
|V12_45011_teacher|settle|10/10|9|0|
|V6_43202|direct|10/10|7|0|
|V6_43202|settle|10/10|9|0|
|V9_44711|direct|10/10|5|0|
|V9_44711|settle|10/10|5|0|
|V9_44711_teacher|direct|10/10|6|0|
|V9_44711_teacher|settle|10/10|6|0|

## flat25 前进 qRMS

首段为 0–1s；settle 的命令在 2s 开始，因此另列命令后的首秒；晚期为 20–25s。

|策略|模式|pass|0–1s peak@t|命令后首秒 peak@t|20–25s peak@t|
|---|---:|---:|---:|---:|---:|
|V9_44711|direct|False|0.4152@0.70s|0.4152@0.70s|0.3625@24.00s|
|V9_44711|settle|False|0.0916@0.02s|0.4167@2.94s|0.3609@25.00s|
|V9_44711_teacher|direct|False|0.4205@0.62s|0.4205@0.62s|0.3669@23.60s|
|V9_44711_teacher|settle|False|0.0917@0.02s|0.4189@2.58s|0.3561@25.00s|
|V6_43202|direct|True|0.1323@0.26s|0.1323@0.26s|0.1319@22.00s|
|V6_43202|settle|True|0.0927@0.02s|0.1395@2.26s|0.1301@20.46s|
|V12_45011|direct|False|0.4643@0.28s|0.4643@0.28s|0.1850@20.18s|
|V12_45011|settle|True|0.1214@0.90s|0.1289@2.52s|0.1664@25.00s|
|V12_45011_teacher|direct|False|0.4638@0.28s|0.4638@0.28s|0.2592@20.78s|
|V12_45011_teacher|settle|True|0.1057@1.00s|0.1313@2.38s|0.1873@24.98s|

## V12 student 前进/后退早期与末段

早期指命令生效后的首秒，末段指 23–25s。转矩与关节速度均为腿部峰值；接触为 base contact 峰。

|模式/方向|区间|实际XY速度均值|高度均值/范围 m|转矩 Nm|关节速度 rad/s|base接触 N|
|---|---:|---:|---:|---:|---:|---:|
|direct/forward|early|0.6420|0.4021/0.0791|120.0000|15.0000|0.0|
|direct/forward|late|0.6621|0.4345/0.0049|28.5546|2.8524|0.0|
|direct/reverse|early|0.4931|0.4517/0.0115|120.0000|15.0000|0.0|
|direct/reverse|late|0.5373|0.4681/0.0004|15.5512|0.0616|0.0|
|settle/forward|early|0.5689|0.4407/0.0106|35.3070|2.3120|0.0|
|settle/forward|late|0.6623|0.4342/0.0007|19.2516|0.1052|0.0|
|settle/reverse|early|0.4286|0.4516/0.0066|45.0659|3.0964|0.0|
|settle/reverse|late|0.3916|0.4610/0.0001|17.3577|0.0603|0.0|

## 关节异常时间与具体失败轨迹

`diagnosis.json` 的 `flat25.forward_qrms_and_joint_timing` 对每个前进 case 给出全部 12 个腿关节的 request、target、actual 峰值时间和两类误差峰；其中 `rl_thigh_joint`、`rr_hip_joint`、`fr_calf_joint` 标记为 emphasized。`normal.failure_trajectories` 给出每个失败 case 的命令后首秒、末2秒、最低高度、qRMS、转矩、速度与接触峰值和发生时间。

`normal.stair_last_2s_jitter` 单独列出每个上下楼梯 case 末2秒的前向速度标准差/峰峰值、高度标准差/峰峰值、跟踪误差、转矩、关节速度、接触和原分析的世界系竖直速度峰。

V12 student 直接前进中，`rl_thigh_joint` 的 actual–target 误差峰为 0.5780rad@0.34s，actual 相对默认姿态峰为 0.5741rad@0.84s；`rr_hip_joint` 的 request–target 间隙峰为 0.6012rad@0.34s、actual–target 误差峰为 0.4884rad@0.30s；`fr_calf_joint` 的 actual–target 误差峰为 0.3780rad@0.06s。settle 前进相应 qRMS 通过，但命令切入附近仍可见 `fr_calf_joint` actual 峰 0.3072rad@2.44s 和 `rl_thigh_joint` actual–target 误差 0.2056rad@2.04s。

V12 student 的具体高风险失败包括：上楼梯 L9@1.2m/s 在 6.14s actuator violation，腿速峰 31.5762rad/s@6.14s；wave L9@0.3m/s 在 0.22s actuator violation，腿速峰 27.4489rad/s；obstacles L6@0.6m/s 在 5.08s illegal contact，base contact 峰 5131.7N；pit 0.90m@0.6m/s 在 3.42s illegal contact，base contact 峰 5192.4N。这些是单个固定轨迹的观测值，不外推为随机发生率。

## 已有图

- `../pilot300_normal/forward_comparison.png` 与 `../pilot300_normal/startup_joints.png`
- `../pilot300_flat25/forward_comparison.png` 与 `../pilot300_flat25/startup_joints.png`
