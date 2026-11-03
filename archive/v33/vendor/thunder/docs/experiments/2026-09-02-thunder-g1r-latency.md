# Thunder G1-R 完整扰动与延迟鲁棒性实验

时间：2026-09-02（Asia/Shanghai）

Task：`DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0`

脚本：`scripts/experiments/run_thunder_g1r_latency.sh`

## 已验证基线

共同固定评估协议为 seed `20260901`、每命令 256 env、预热 200 步、采集 600 步、`Kp/Kd=90/6.93`，关闭评估随机化。

| 模型 | 已验证结果 | 结论 |
| --- | --- | --- |
| `model_13500.pt` | 六运动 tracking RMSE `0.05374`；静止漂移 `0.000143 m/s`；全场景接触峰值 `262.80 N` | 当前静止与接触基线 |
| Post-20k S4 `model_7996.pt` | tracking RMSE `0.04375`；静止漂移 `0.005010 m/s`；接触峰值 `288.07 N` | 负载课程提升平均跟踪，但静止变差 |
| S4-G1 `model_999.pt` | 左右横移实际 `+0.5875/-0.5926 m/s`；接触峰值 `272.15/286.14 N`；静止漂移 `0.003303 m/s` | 本实验唯一训练起点；横移速度和落脚均优于 G2 |
| S4-G2 `model_4998.pt` | 左右横移实际 `+0.5520/-0.5496 m/s`；接触峰值 `616.53/597.57 N` | 已失败，不再续接 |

S4-G1 起点完整路径：

```text
logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_23-06-48_lateral_smooth_s4_7996_g1_v2/model_999.pt
```

## 待训练与验证

所有新分支使用 `4,096 envs`、seed `42`，从起点只 warm-start actor 和 learned std，critic、optimizer 和 iteration 重置。

### 历史 G1R-A / G1R-B（B 已作废）

以下是已执行的历史对照，不再作为可启动训练方案。两个分支原计划各训练到 5,000 iterations；命令比例、负载和随机化保持一致。

- `G1R-A`：`joint_mirror=-0.10`。
- `G1R-B`：**已作废并停止**。其纯 yaw scale `0.5`、纯横移 scale `0.0` 不符合全模式对称要求。
- 两支均保持 G1 的 orientation、hip limit、action rate、加速度、slip、clearance、minimum contacts 和 touchdown 权重。
- 若两支均未达到横移水平门槛，可从优胜 checkpoint 运行 `orientation-recovery`，仅将 `flat_orientation_l2` 从 `-1.0` 调到 `-1.5`，追加 1,000 iterations。

旧名义入口已从可执行模式中移除。当前只允许从 A 5k 启动全模式对称恢复：

```bash
bash scripts/experiments/run_thunder_g1r_latency.sh symmetric-recovery
```

`nominal-select` 对 A/B 的 5k 最终 checkpoint 做完整名义评估。只有通过全部名义硬门槛的候选才能晋级；多个候选合格时依次按六运动 tracking RMSE、横移姿态 RMS/峰值、接触峰值选择，并将绝对 checkpoint 写入：

```text
outputs/thunder_g1r_latency_20260902/nominal_winner.txt
```

### LAT-OBS / LAT-DROP / LAT-ACT

三个敏感性分支从同一个名义优胜 checkpoint 各训练 500 iterations：

- `LAT-OBS`：每环境、每回合采样 0–3 帧物理观测延迟。
- `LAT-DROP`：每步 2% 概率触发 1–3 帧丢帧 burst，保持最后有效观测。
- `LAT-ACT`：每环境、每回合采样 0–3 帧动作 FIFO 延迟。

```bash
bash scripts/experiments/run_thunder_g1r_latency.sh latency-probes
```

### LAT-COMBINED

组合分支仍从名义优胜 checkpoint 开始，随后完整 resume：

| 阶段 | 训练量 | 观测/动作最大延迟 | 丢帧 burst |
| --- | ---: | ---: | ---: |
| `lat_combined_1f` | 1,000 | 1 帧 / 20 ms | 1 帧 |
| `lat_combined_2f` | 1,000 | 2 帧 / 40 ms | 1–2 帧 |
| `lat_combined_3f` | 2,000 | 3 帧 / 60 ms | 1–3 帧 |

```bash
bash scripts/experiments/run_thunder_g1r_latency.sh latency-combined
```

## 验收与记录

每个长训阶段先用起点 checkpoint 运行 32 步 smoke，finite 或异常终止检查失败即停止。每 500 iterations 仅做轻量筛查：payload `0/12 kg`、静止/前进/左横移/左 yaw、64 env、预热 100 步、采集 200 步；阶段最终 checkpoint 才执行 payload `0/6/12 kg`、完整七命令、256 env、预热 200 步、采集 600 步。

`lat_combined_3f` 最终 checkpoint 额外执行固定 `0/1/2/3` 帧的 OBS、DROP、ACT、COMBINED 矩阵。OBS/ACT 使用固定帧数；DROP 受现有接口约束，表示 2% 触发概率下 `1..N` 帧 burst 上限。组合三帧结果必须相对零延迟 tracking RMSE `≤1.20×`、接触峰值 `<450 N`、横移姿态峰值 `<2°`、finite 且无异常终止。

名义优胜模型和 `lat_combined_3f` 还会分别运行命令 step/ramp；JSON/CSV 记录 raw policy action 与实际进入 action manager 的 applied action 的一阶变化和二阶差分 RMS/峰值，用于定位连续命令下的动作突变。

训练日志、结果和 manifest 分别写入：

```text
outputs/thunder_g1r_latency_20260902/logs/
outputs/thunder_g1r_latency_20260902/results/
outputs/thunder_g1r_latency_20260902/manifests/stages.ndjson
outputs/thunder_g1r_latency_20260902/progress.log
```

名义硬门槛：六个运动命令分别不超过 S4-G1 对应 RMSE 的 `1.05×`，横移实际速度绝对值 `≥0.58 m/s`，左右 yaw `≥0.57 rad/s`，横移 roll/pitch RMS `<0.5°` 且峰值 `<1°`，接触峰值目标 `≤300 N`、硬上限 `<450 N`，静止漂移 `≤0.002 m/s`，静止腿/hip RMS 不高于 G1 的 `7.486/2.247 Nm`，腿/轮超过额定扭矩 90% 的比例 `≤1%`，无 NaN 或异常终止。另记录相对 `model_13500` 静止腿/hip RMS 的倍率。

延迟硬门槛：零延迟相对名义模型恶化不超过 5%；三帧组合延迟的平均 tracking RMSE 不超过零延迟 `1.20×`；接触峰值 `<450 N`、横移姿态峰值 `<2°`，无跌倒、NaN 或异常动作突变。

当前状态（2026-09-02 05:11 CST）：**G1R-A 已从 `model_300.pt` 完整恢复并完成等效 2,000 iterations，checkpoint 为 `2026-09-02_04-27-24_g1r_a_recover_2k/model_1998.pt`。** 结束时在线随机分布误差约为 XY `0.321`、yaw `0.413`，仅作为健康监控，不能替代固定评估。B 首次随 A 串行启动时在 Isaac 初始化阶段收到外部 SIGINT，未消耗迭代；拆成独立 `nominal-b` 后 32 步 smoke 已通过并于 05:11 重新训练。远端队列已设置为 B 2k → A/B 各追加 3k → 严格选模 → 延迟 probes → 通过选模后再跑 4k combined。未通过全部硬门槛前，继续保留 S4-G1 `model_999.pt` 作为推荐模型。

2026-09-02 10:34 CST 更新：G1R-B 因纯横移关闭 mirror、纯 yaw 仅保留 0.5 倍 mirror，与“所有运动模式保持对角腿对称”的目标冲突，已在保留 checkpoint 后停止；依赖 B 的选模和延迟队列同步取消。新的 `g1r_symmetric_recovery_8k` 从 G1R-A 5k `model_4997.pt` 完整恢复，追加 3,000 iterations：`joint_mirror=-0.15` 且四种模式 scale 全为 `1.0`，同时针对 A 的已验证缺口将 `flat_orientation=-1.5`、静止平移/yaw/轮速惩罚调为 `-4.0/-1.0/-0.02`，yaw tracking 调为 `2.25`。32 步 smoke 已通过，10:38 CST 正式启动。

## 已锁定的训练需求

- Flat 同时覆盖静止、前后、左右横移、左右 yaw、混合命令及连续 step/ramp 命令变化。
- 纯静止要低漂移、低腿/hip 内力；横移和 yaw 要跟踪准确，身体 roll/pitch 尽量水平。
- 所有运动模式保留腿部对称约束；纯 yaw 额外约束四轮协调，不以牺牲横移为代价。
- 轮足必须主动摆腿、减少拖地和蹩脚，同时控制 touchdown、峰值接触力、wheel slip 和接触占空比。
- 动作、关节速度/加速度和扭矩变化要平滑；降低腿/hip torque、轮 torque 与机械功率，避免负载转移和执行器饱和。
- 保持 `Kp/Kd=90/6.93` 主线并记录 Kp/Kd 对照；保持 53/56 维观测和 16 维动作部署接口。
- 完整随机化覆盖 `0–12 kg` payload、质量/COM、摩擦、执行器 gain、外力/外矩、push、reset 姿态和六维速度。
- 延迟鲁棒性后续覆盖 0–3 帧观测延迟、观测丢帧和动作输出延迟，分别训练后再做组合课程。
- Rough terrain 独立训练，不与本轮 Flat Yaw 专项混用；每个任务记录起点、参数、时间、checkpoint、固定评估和实机待验证项。

## S4-G2 纯 Yaw 特训

S4-G2 `model_4998.pt` 的左右 yaw 为 `+0.6073/-0.5901 rad/s`，对应 tracking RMSE `0.0346/0.0250`，是当前历史最佳 yaw；但它的左右横移接触峰值为 `616.53/597.57 N`，不能作为完整运动模型发布。G2 的 yaw 优势主要来自更长的协调精修和较强的姿态/对称/平滑约束；横移失败主要与当时 `clearance=-100`、`slip=-0.20`、`hip limit=-20` 等激进组合同时出现。因此本分支只 warm-start G2 actor/std，重置 critic 与 optimizer，并保留轻落脚版本的奖励上限。

`yaw_specialization_g2_3k` 使用 `4,096 envs`、seed `42`、3,000 iterations、完整 `0–12 kg` payload 与 S4 随机化。命令比例为静止/纯 yaw/纯横移/混合 `10/50/5/35%`；`joint_mirror=-0.20` 且所有模式 scale 均为 `1.0`；新增纯 yaw 四轮原始转速方差惩罚 `-0.002`。Yaw tracking 调为 weight `2.5`、std `0.25`，纯 yaw 平移漂移 `-3.0`，姿态/roll-pitch 角速度 `-2.0/-0.10`，yaw 摆腿高度目标 `0.04 m`、权重 `-8.0`，touchdown `-0.3`。接触力阈值仍为 `450 N`，没有恢复 G2 的激进 clearance/slip/hip-limit 参数。

2026-09-02 11:47 CST：全模式对称恢复分支保留到 iteration `5981/7997` 后停止，最后在线 mean reward `307.11`；该值不是固定命令评估。纯 Yaw 特训已启动，先执行 G2 起点的 32 步 smoke，输出写入 `outputs/thunder_yaw_specialization_20260902/`。最终仍必须同时评估静止、前后、横移、yaw、接触力和姿态；Yaw 提升不能抵消横移重踩。

2026-09-02 约 14:00 CST：`yaw_specialization_g2_3k` 完成 `model_2999.pt`。下一段 `yaw_soft_touchdown_2k` 只 warm-start其 actor/std并重置critic/optimizer，保持 `4 cm` Yaw clearance和 `clearance=-8`，把 touchdown从 `-0.3`提高到 `-0.8`、允许下落速度从 `0.15`收紧到 `0.10 m/s`，并把接触力惩罚起点从 `450 N`降至软目标 `300 N`；评估硬上限仍为 `450 N`。本段不增加新的奖励函数，避免再次混淆抬腿和落地效果。

2026-09-02 固定评估裁决：`yaw_specialization/model_2999.pt` 不晋级。空载左右Yaw只有 `+0.547/-0.554 rad/s`，左Yaw接触峰值 `511.6 N`，静止漂移约 `0.008 m/s`；6kg最大峰值 `565.3 N`，12kg最大峰值 `606.1 N`。它相对父模型S4-G2 `model_4998.pt` 同时损失Yaw跟踪和接触安全，因此禁止部署，也不再作为默认父模型。

`yaw_soft_touchdown_2k` 已错误继承上述失败父模型。2026-09-02 16:06 CST在 iteration `1151/2000` 停止，最后完整checkpoint为 `model_1100.pt`；所有产物保留并标记 `STOPPED_INVALID_PARENT`，不删除、不续训、不部署。今后执行血统门控：失败父模型的后代只有在独立完整固定评估通过后才能重新取得晋级资格；未评估或只看在线reward一律无资格。

### 当前有效继承路线

```text
fs20k/model_13500                         综合名义基线，可做受保护实机测试
├─ smooth-from-13500 G1/G2                最终G2重踩，整支不晋级
└─ future nominal-soft-contact            应从这里重新开始，尚未训练

fs20k/model_19999                         固定评估失败
└─ Post-20k S1→S2→S3→S4/model_7996       负载研究完成，但未通过综合门槛
   ├─ S4-G1/model_999                     横移/轻落脚诊断父模型
   │  ├─ G1R-A/B                          均失败
   │  └─ symmetric recovery               未固定评估，已停止
   └─ S4-G2/model_4998                    仅Yaw专项诊断可用，禁止横移部署
      └─ yaw specialization/model_2999    固定评估失败
         └─ yaw soft touchdown/model_1100  无效后代，已停止

Rough V0                                  从零独立训练，不继承上述Flat模型
Latency                                   尚未训练；未来只能从通过名义门槛的父模型启动
```
