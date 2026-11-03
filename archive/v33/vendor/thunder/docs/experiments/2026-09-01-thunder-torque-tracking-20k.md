# Thunder Flat：20k 全随机化降扭矩与跟踪训练记录

## 需求与目标

- 每个正式对照从随机初始化开始，不加载 `model_11994.pt`，不使用 `--resume`。
- 每组训练 `20,000 iterations`、`32,768 envs`、训练 seed `42`。
- 保留全部已有 domain randomization，包括执行器增益、接触材质、质量、COM、外力、reset 与 push。
- Flat 保持平面几何；足端碰撞材质随机化使有效地面接触静摩擦为 `0.3–1.0`、动摩擦为 `0.3–0.8`、恢复系数为 `0–0.5`（64 buckets）。不在本轮混入 rough 几何。
- 目标是降低腿部实际 `applied_torque` mean/RMS，改善前后、横移和 yaw 跟踪，同时不把负载转移到轮子，并保持接触历史峰值不超过 `450 N`。

## 已排除的旧实验

`p1_pose_20` 从 `model_11994.pt` 续训 400 iterations，只验证 `|hip|=0.20` 姿态，并把 Kp/Kd 随机化固定为 `1.0`。该实验不满足本轮“从零 + 全随机化 + 20k”的要求，已停止，结果不进入正式比较。

## 本轮固定训练条件

- Task：`DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0`
- 初始姿态：`|hip|=0.20 rad`、base `z=0.46 m`，其余 thigh/calf 保持 Thunder 默认姿态。
- 命令分布：静止 10%、纯 yaw 10%、纯横移 10%、其余混合运动 70%。
- 执行器增益随机化：每次 reset 对 Kp/Kd 分别按 uniform `0.5–2.0×` 缩放。
- base 质量：增加 `-1–3 kg`；其他刚体质量：`0.7–1.3×`；base COM：x/y/z 各 `±0.05 m`。
- reset 外力：各轴 `±30 N`；力矩：各轴 `±10 Nm`。
- base reset：位置、姿态和六维速度使用环境默认范围；每 `10–15 s` 施加 x/y `±0.5 m/s` push。
- 轻触地奖励保持：接触力阈值 `450 N`、权重 `-1e-2`；touchdown velocity 权重 `-0.2`。

## 自动训练队列

| 顺序 | 运行名 | Kp/Kd | 腿扭矩权重 | 线速/角速跟踪权重 | 起点 |
| ---: | --- | --- | ---: | --- | --- |
| 1 | `fs20k_kp90_tw3e5_track125` | `90/6.93` | `-3e-5` | `3.75/1.875` | 从 0 |
| 2 | `fs20k_kp120_tw3e5_track125` | `120/8` | `-3e-5` | `3.75/1.875` | 从 0 |
| 3 | `fs20k_kp60_tw3e5_track125` | `60/5.66` | `-3e-5` | `3.75/1.875` | 从 0 |
| 4 | `fs20k_baseline_kp120_tw1e5_track100` | `120/8` | `-1e-5` | `3.0/1.5` | 从 0 |

所有组除了表中三项外配置一致。首组优先运行，是为了尽早得到预计更平衡的可用模型；第四组提供同训练预算的从零基线。

## 基准与验收口径

旧 `model_11994.pt` 固定评估基准：六个运动命令平均腿扭矩 mean(abs) `12.007 Nm`、RMS `15.050 Nm`，横移/yaw hip RMS `23.577 Nm`，六命令平均 tracking RMSE `0.07404`。

新模型统一用固定评估 seed `20260901`，对静止、`vx=±0.6 m/s`、`vy=±0.6 m/s`、`yaw=±0.6 rad/s` 各采集 256 env、预热 200 步、记录 600 步。主要目标：腿 torque mean/RMS 下降至少 10%，横移/yaw hip RMS 下降至少 10%，平均 tracking RMSE 改善至少 5%，任一命令不劣化，接触历史峰值不超过 `450 N`。

## 输出与提升记录

- 总进度：`outputs/thunder_20k_full_dr_20260901/progress.log`
- 每组训练日志：`outputs/thunder_20k_full_dr_20260901/logs/train_<run>.log`
- 每组评估日志：`outputs/thunder_20k_full_dr_20260901/logs/evaluate_<run>.log`
- 每组完整指标：`outputs/thunder_20k_full_dr_20260901/results/<run>.json` 与 CSV。

每组完成后先记录 checkpoint，再立即做同口径评估；只有完整指标证明提升，才会写入“已达成”，不以训练 reward 单独宣称成功。

## 当前进度

- `03:27`：首次启动只到配置解析；Isaac Sim 首次 EULA 和 Hydra 对整数 Kp 类型的校验先后阻断，尚未产生训练 iteration，已修正且保留失败记录。
- `03:29`：`fs20k_kp90_tw3e5_track125` 正式从 0 启动；确认命令没有 resume/checkpoint，也没有 actuator gain DR 覆盖。
- `03:31`：已到 iteration `15/20000`，单轮约 `1.96–2.27 s`；现场采样 GPU 利用率约 `68%`、显存约 `16.5/97.9 GB`。早期 ETA 约 `12 h 20 min`，待速度稳定后再更新。

## 阶段性评估：model_13500

在首组仍训练时，用 `model_13500.pt` 做固定 seed `20260901` 的名义参数评估。每个命令 256 env，预热 200 步、采样 600 步；该快照不是最终 20k 结论。

| 汇总指标 | model_11994 | model_13500 | 变化 |
| --- | ---: | ---: | ---: |
| 六运动命令 tracking RMSE | 0.07404 | 0.05375 | -27.4% |
| 腿 mean(abs torque) | 12.007 Nm | 9.093 Nm | -24.3% |
| 腿 torque RMS | 15.050 Nm | 11.163 Nm | -25.8% |
| 腿 P99（六命令均值） | 40.981 Nm | 25.036 Nm | -38.9% |
| 腿全局最大扭矩 | 62.028 Nm | 38.455 Nm | -38.0% |
| 横移+yaw hip RMS | 23.577 Nm | 14.594 Nm | -38.1% |
| 轮 torque mean | 2.204 Nm | 2.224 Nm | +0.9% |
| 轮 torque RMS | 2.841 Nm | 2.843 Nm | +0.04% |
| 腿功率 mean | 5.453 W | 4.122 W | -24.4% |
| 腿功率 RMS | 8.946 W | 6.222 W | -30.4% |
| 轮功率 mean | 8.734 W | 8.050 W | -7.8% |
| 全局接触峰值 | 409.78 N | 263.07 N | -35.8% |

逐命令主要速度：前进 `0.5691→0.5766`，后退 `-0.5926→-0.5908`，左横移 `0.5608→0.5634`，右横移 `-0.5780→-0.5693`，左 yaw `0.4686→0.5626`，右 yaw `-0.5014→-0.5436`。总体改善主要来自 yaw；后退 RMSE 增加 23.2%、左横移增加 1.5%、右横移增加 8.7%，因此尚未通过“任何命令不得变差”。

六个命令的腿 torque mean 全部下降 `17.6–35.2%`，横移/yaw hip RMS 全部下降 `32.0–45.8%`。但左横移轮 torque RMS 增加 20.2%，右横移增加 10.5%；左 yaw 轮功率 mean 增加 31.0%。因此总体降扭矩已成立，但“不得把负载转移到轮子”尚未逐命令成立。

接触方面：左横移峰值 `409.78→245.64 N`，右横移 `249.01→263.07 N`，左 yaw `262.05→245.58 N`，右 yaw `253.51→223.24 N`；全部低于 450 N。所有命令 unexpected done 为 0，训练 episode length 为 1000/1000，time-out 为 1.0、出界为 0。

阶段性判断：aggregate 扭矩、hip、tracking、功率和接触峰值均超过目标，但右横移/后退跟踪和横移轮负载仍是明确短板。等待 20k checkpoint 后再做正式排名，不用阶段性结果替换最终配置。

## model_11994 到 model_13500 的配置变化

保存的 `env.yaml`/`agent.yaml` 表明两者只有以下实质差异：

| 项目 | model_11994 | model_13500 | 意图 |
| --- | ---: | ---: | --- |
| 腿 Kp | 120 | 90 | 降低控制带宽和动态峰值 |
| 腿 Kd | 8 | 6.93 | 按 `8×sqrt(90/120)` 保持相近阻尼比 |
| `joint_torques_l2` | `-1e-5` | `-3e-5` | 三倍强化腿扭矩平方惩罚 |
| 线速度跟踪权重 | 3.0 | 3.75 | 防止策略以减速换低扭矩 |
| yaw 跟踪权重 | 1.5 | 1.875 | 同比例强化 yaw 跟踪 |
| 训练方式 | 从 `model_11495` 续训 500 轮 | 从 0 独立训练 13,500 轮 | 新模型全程使用同一目标 |

`model_11994` 的上一个阶段接触峰值惩罚仍为 `-0.002`，最后 500 轮才改成当前 `-0.01`；新模型从第 0 轮就使用 `-0.01` 和 touchdown `-0.2`。这会显著增加新模型对轻触地目标的一致训练时间。

未变化项：`|hip|=0.20` 姿态、base z、32768 env、seed 42、10% 静止/10% 纯 yaw/10% 纯横移/70% 混合命令、全部 domain randomization、Flat 平面与材质范围、动作/观测顺序、网络结构、PPO 参数、450 N 阈值及其他奖励。

当前实验一次同时改变了 Kp/Kd、扭矩权重、跟踪权重和训练起点，因此不能仅凭 model_13500 将全部提升归因于某一个参数。队列中的从零 baseline、Kp120 和 Kp60 分支会分别拆开训练起点、奖励和控制带宽的贡献。

## model_16800 固定命令视频

使用相同的 `model_16800.pt`、名义 `Kp/Kd=90/6.93`、无随机化和 push，分别录制 5 秒的 `±0.6 m/s` 前后/横移和 `±0.6 rad/s` yaw：

| 命令 | 实际主轴速度 | roll 最大值 | pitch 最大值 | 高度峰峰值 |
| --- | ---: | ---: | ---: | ---: |
| 前进 | 0.5815 m/s | 0.25° | 0.31° | 0.08 mm |
| 后退 | -0.6000 m/s | 0.05° | 0.14° | 0.35 mm |
| 左横移 | 0.5695 m/s | 0.80° | 3.12° | 7.05 mm |
| 右横移 | -0.5654 m/s | 0.71° | 2.74° | 6.02 mm |
| 左 yaw | 0.5727 rad/s | 0.41° | 0.69° | 9.05 mm |
| 右 yaw | -0.5239 rad/s | 0.50° | 1.49° | 11.51 mm |

纯 yaw 没有明显扭身：左右 yaw 的 roll 最大值均不超过 0.50°，pitch 最大值不超过 1.49°。右 yaw 的跟踪与高度波动弱于左 yaw，仍是后续需要改善的一侧。视频与逐段 JSON 位于 `outputs/thunder_20k_full_dr_20260901/videos_model_latest/`。

## 正式固定评估：model_18800 与 model_19999

首组 `fs20k_kp90_tw3e5_track125` 完成 20,000 iterations 后，按与 `model_13500` 完全相同的协议复评 `model_18800` 和最终 `model_19999`：`pose=0.20`、`Kp/Kd=90/6.93`、seed `20260901`、每命令 256 env、预热 200 步、采样 600 步，并关闭 observation corruption、push、质量、COM、材质、外力和 gain randomization。

| 汇总指标 | model_13500 | model_18800 | 相对 13500 | model_19999 | 相对 13500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 六运动命令 tracking RMSE | 0.05375 | 0.06347 | +18.1% | 0.05719 | +6.4% |
| 腿 mean(abs torque) | 9.093 Nm | 8.975 Nm | -1.3% | 9.203 Nm | +1.2% |
| 腿 torque RMS | 11.163 Nm | 10.938 Nm | -2.0% | 11.135 Nm | -0.3% |
| 腿 P99（六命令均值） | 25.036 Nm | 24.463 Nm | -2.3% | 24.302 Nm | -2.9% |
| 横移+yaw hip RMS | 14.594 Nm | 14.442 Nm | -1.0% | 14.357 Nm | -1.6% |
| 轮 torque RMS | 2.843 Nm | 3.146 Nm | +10.7% | 3.155 Nm | +11.0% |
| 腿功率 RMS | 6.222 W | 5.880 W | -5.5% | 6.126 W | -1.5% |
| 轮功率 RMS | 10.383 W | 11.308 W | +8.9% | 11.993 W | +15.5% |
| 全局接触峰值 | 263.07 N | 268.58 N | +2.1% | 256.74 N | -2.4% |
| 静止前向漂移 | 0.00004 m/s | 0.00401 m/s | — | 0.01333 m/s | — |

`model_19999` 相对 `model_13500` 的逐命令 tracking RMSE 变化为：前进 `+48.9%`、后退 `-9.1%`、左横移 `-10.4%`、右横移 `-11.0%`、左 yaw `+18.4%`、右 yaw `+16.9%`。对应主轴实际速度为前进 `0.5660 m/s`、后退 `-0.5917 m/s`、左横移 `0.5562 m/s`、右横移 `-0.5635 m/s`、左 yaw `0.5461 rad/s`、右 yaw `-0.5342 rad/s`。

`model_18800` 不是更好的中间点：它的前进达到 `0.6020 m/s`，但右横移 tracking RMSE 增加 `20.1%`，左/右 yaw 分别增加 `64.6%/29.9%`，六命令平均恶化 `18.1%`。

按既定归一化分数（越低越好），`model_18800=1.0235`、`model_19999=1.0110`，均差于 `model_13500=1.0`。两者都未通过临时门槛；共同问题是 yaw 跟踪与轮 torque RMS，最终模型还额外违反轮功率 mean/RMS/P99 门槛。

正式结论：首组训练在 `model_13500` 后没有继续产生净提升。后段训练只带来很小的腿 P99、hip RMS 和接触峰值改善，却把负载转移到轮子，并破坏前进、yaw 和纯静止表现。因此保留 `model_13500` 作为当前已验证最佳 checkpoint，不采用 `model_18800` 或 `model_19999`，也不从最终 checkpoint 继续续训。

完整原始结果：

- `outputs/thunder_20k_full_dr_20260901/interim/fs20k_kp90_model13500.json`
- `outputs/thunder_20k_full_dr_20260901/results/model_18800_fixed.json`
- `outputs/thunder_20k_full_dr_20260901/results/model_19999_fixed.json`

## 后续对照训练状态

第二组 `fs20k_kp120_tw3e5_track125` 已从 0 开始训练，除 `Kp/Kd=120/8` 外与首组一致。2026-09-01 15:22 CST 采样时为 iteration `1110/20000`，32,768 env，mean episode length `1000`，预计尚需约 10.5 小时。该组完成前不以训练 reward 判断优劣，完成后使用同一固定评估协议与 `model_13500` 比较。

## Post-20k：静止、纯 yaw 与负载优化

- 旧 Kp120 队列于 2026-09-01 安全停止，最后完整 checkpoint `model_2600.pt` 保留，并记录为 `STOPPED_FOR_POST20K`；Kp60 未启动。
- 新训练固定 `32,768 env`、seed `42`、`Kp/Kd=90/6.93`，命令占比调整为静止 `15%`、纯 yaw `15%`、纯横移 `10%`、混合 `60%`。
- 新增静止平移/yaw/轮速惩罚、纯 yaw 横向漂移惩罚，以及轮端 torque/power 惩罚；删除不符合静止语义的 `feet_contact_without_cmd`。
- S1 从 `model_19999.pt` 仅 warm-start actor 和 learned std；critic、optimizer、iteration 全部重置，学习率固定 `2e-4`。S2–S4 分别引入 `0–4/8/12 kg` 点质量负载并完整恢复上一阶段。
- 首次启动暴露 Hydra 整数/浮点类型错误；第二次暴露 RSL-RL 3.1.2 `load_state_dict()` 返回布尔值的兼容差异。两处均在产生训练 iteration 前修复，失败日志保留。
- 2026-09-01 16:38 CST 第三次启动成功。日志确认仅加载 `8` 个 actor 参数张量和 `std`，`8` 个 critic 参数保持全新；iteration 从 `0/1000` 开始。
- iteration `20/1000`：`1.87 s/iter`、约 `420k steps/s`、GPU 约 `16.3/97.9 GB`、利用率约 `64%`。S1 轮 torque 奖励率 `-0.0298`、轮 power `-0.0007`，与预期量级一致；无出界。

持续状态与阶段结果写入 `outputs/thunder_post20k_payload_20260901/`；screen 会话为 `doso-post20k`，主启动日志为 `/tmp/doso-post20k-launch.log`。

### S1 边界结果与 S2 启动

- S1 于 2026-09-01 17:13 CST 完成，checkpoint 为 `2026-09-01_16-38-49_s1_nominal_repair/model_999.pt`；随后 S2 从该 checkpoint 完整恢复 actor、critic 和 optimizer，继续训练 `0–4 kg` payload。
- 首次 S1 筛查因 Isaac Lab 2.3.2 函数名应为 `euler_xyz_from_quat` 而失败；训练未受影响。修复后补跑 `model_999` 空载、固定 seed、四命令筛查。
- 三个运动命令（前进、左横移、左 yaw）平均 tracking RMSE：`0.05684`，比 `model_19999` 的 `0.06160` 改善 `7.7%`，但比 `model_13500` 的 `0.05595` 差 `1.6%`。
- 静止平移漂移：`0.00795 m/s`，比 `model_19999` 改善 `40.4%`，但仍远高于 `model_13500` 的 `0.000139 m/s` 和硬门槛 `0.002 m/s`。
- 三运动命令平均腿 RMS：`13.44 Nm`，比 `model_19999` 高 `21.4%`；轮 RMS `2.90 Nm`，比 `model_19999` 低 `7.7%`；轮功率 RMS `11.75 W`，比 `model_19999` 高 `2.7%`。
- 接触峰值 `340.02 N`，低于安全上限 `450 N`，但未达到空载门槛 `263.07 N`。所有筛查命令 unexpected done 为 `0`。

S1 的结论是：跟踪和静止相对 `model_19999` 已有回升，轮扭矩下降，但腿扭矩、轮功率、静止漂移和接触峰值仍未通过发布门槛；按既定规则继续 S2，不晋级 S1 为默认模型。

### Post-20k 阶段完成与 checkpoint 编号说明

Post-20k 于 2026-09-01 21:11 CST 完成，共追加 `8,000 iterations`。S1 只从 `model_19999.pt` warm-start actor 和 learned std；S2-S4 完整恢复上一阶段的 actor、critic、optimizer 和 iteration。

| 阶段 | 新增 iterations | Payload | 零载比例 | 输入 checkpoint | 输出 checkpoint | 等效累计 |
| --- | ---: | --- | ---: | --- | --- | ---: |
| S1 空载修复 | 1,000 | `0 kg` | 100% | `model_19999.pt` | `s1_nominal_repair/model_999.pt` | 21k |
| S2 轻载 | 2,000 | `0-4 kg` | 50% | S1 `model_999.pt` | `s2_payload_4kg/model_2998.pt` | 23k |
| S3 中载 | 2,500 | `0-8 kg` | 40% | S2 `model_2998.pt` | `s3_payload_8kg/model_5497.pt` | 25.5k |
| S4 满载 | 2,500 | `0-12 kg` | 35% | S3 `model_5497.pt` | `s4_full_payload/model_7996.pt` | 28k |

因此，最终 checkpoint `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_19-44-59_s4_full_payload/model_7996.pt` 在训练血统上等效于从 `model_19999` 再训练约 8k，即口头所称的 `model_27999`/28k。文件编号从 Post-20k 新链路重新计数，且 RSL-RL 最终保存存在少量编号偏差；完整训练量以 `progress.log` 的 `POST20K_COMPLETE total_iterations=8000` 为准。

运行真源：

- 阶段进度：`outputs/thunder_post20k_payload_20260901/progress.log`
- 阶段 manifest：`outputs/thunder_post20k_payload_20260901/manifests/stages.ndjson`
- 训练日志：`outputs/thunder_post20k_payload_20260901/logs/train_s*.log`
- 评估结果：`outputs/thunder_post20k_payload_20260901/results/`

### 侧移与纯 yaw 平滑训练：双基线对照

目的：两条分支执行完全相同的 G1/G2 训练课程；分支之间唯一实验变量是 actor 和 learned std 的策略初始化，分别来自已验证跟踪较好的 `model_13500`，以及经过 Post-20k 负载课程的等效 `model_27999`（实际文件为上述 `s4_full_payload/model_7996.pt`）。

两条分支共同使用 Flat Thunder、`4,096 envs`、seed `42`、`Kp/Kd=90/6.93`、固定学习率 `2e-4`。命令每 10 秒重采样：静止 10%、纯 yaw 20%、纯横移 30%、混合运动 40%；平移/yaw tracking 权重为 `3.75/1.875`。

| 配置 | G1 | G2 |
| --- | --- | --- |
| 训练量 | 1,000 iterations | 完整恢复 G1 后追加 4,000 iterations |
| 恢复方式 | 只 warm-start actor/std，重置 critic/optimizer | 完整 resume actor/critic/optimizer/iteration |
| Payload | `0-2 kg`，70% 零载 | `0-4 kg`，50% 零载 |
| 外力/外矩 | `±5 N / ±1.5 Nm` | `±10 N / ±3 Nm` |
| Push | `±0.2 m/s` | `±0.3 m/s` |
| `flat_orientation_l2` | `-1` | `-2` |
| `hip_abs_limit_l2` | `-5` | `-20` |
| `joint_mirror` | `-0.10` | `-0.15` |
| `action_rate_l2` | `-0.015` | `-0.02` |
| 腿/轮加速度 | `-3e-7 / -5e-10` | `-5e-7 / -1e-9` |
| 轮地滑动 | `-0.05` | `-0.20` |
| 摆动高度 | `-5` | `-100` |
| 最小接触轮数 | `-0.5` | `-1` |

| 分支 | actor 起点 | 状态（2026-09-02 00:33 CST） | checkpoint/进度真源 |
| --- | --- | --- | --- |
| `lateral_smooth_g1_v2 -> g2_v2` | `model_13500.pt` | 已完成 5,000 iterations | 最终 `2026-09-01_22-02-31_lateral_smooth_g2_v2/model_4998.pt`；`outputs/thunder_lateral_smooth_4096_20260901/progress.log` |
| `lateral_smooth_s4_7996_g1_v2 -> g2_v2` | 等效 `model_27999` 的 `S4/model_7996.pt` | 已完成 5,000 iterations；训练日志最后一轮在线健康指标为 XY/yaw error `0.1796/0.2306`、timeout `1.0`、无出界，正式固定评估待补 | 最终 `2026-09-01_23-27-27_lateral_smooth_s4_7996_g2_v2/model_4998.pt`；`outputs/thunder_flat_post20k_s4_7996_lateral_smooth_4096_20260901/progress.log` |

第一版 wheel contact geometry 判定有误，旧 G2 在产生可用结论前停止并保留 `STOPPED_INVALID_WHEEL_CONTACT_GEOMETRY` 记录；两条正式对照均使用修正后的 `v2`。训练 reward 和在线 tracking error 只用于健康监控，最终是否提升必须由相同固定命令评估给出。

可复现路径：

- 统一训练脚本：`scripts/experiments/run_thunder_lateral_recovery.sh`
- `model_13500` 起点：`logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_03-29-35_fs20k_kp90_tw3e5_track125/model_13500.pt`
- `model_13500` G1：`logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_21-45-20_lateral_smooth_g1_v2/model_999.pt`
- `model_13500` 最终：`logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_22-02-31_lateral_smooth_g2_v2/model_4998.pt`
- 等效 `model_27999` 起点：`logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_19-44-59_s4_full_payload/model_7996.pt`
- 等效 `model_27999` G1：`logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_23-06-48_lateral_smooth_s4_7996_g1_v2/model_999.pt`
- 等效 `model_27999` 最终：`logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_23-27-27_lateral_smooth_s4_7996_g2_v2/model_4998.pt`

### 侧移平滑固定评估结果

2026-09-02 00:56-00:59 CST 使用当前评估器重跑四个 checkpoint。协议固定为 seed `20260901`、7 个命令、每命令 256 env、预热 200 步、采样 600 步、pose `0.20`、`Kp/Kd=90/6.93`；关闭 observation corruption、质量/COM、外力、push、材质和增益随机化。

| 汇总指标 | `model_13500` | Post-20k S4 `model_7996` | 平滑自 `model_13500` | 平滑自 S4 `model_7996` |
| --- | ---: | ---: | ---: | ---: |
| 六运动 tracking RMSE | 0.05374 | 0.04375 | 0.03732 | **0.03711** |
| 腿 torque mean | 9.093 Nm | 9.926 Nm | 8.812 Nm | **8.541 Nm** |
| 腿 torque RMS | **11.163 Nm** | 12.671 Nm | 11.441 Nm | 11.243 Nm |
| 横移+yaw hip RMS | **14.594 Nm** | 18.374 Nm | 14.770 Nm | 14.735 Nm |
| 轮 torque RMS | 2.842 Nm | 1.337 Nm | **1.291 Nm** | 1.404 Nm |
| 腿 power RMS | 6.222 W | 7.024 W | **5.439 W** | 5.543 W |
| 轮 power RMS | 10.383 W | 4.292 W | **3.763 W** | 4.694 W |
| 横移+yaw 接触占空比 | 0.892 | 0.841 | **0.626** | 0.627 |
| 全场景接触峰值 | **262.80 N** | 288.07 N | 573.41 N | 616.53 N |
| 静止平移漂移 | **0.000143 m/s** | 0.005010 m/s | 0.007889 m/s | 0.000599 m/s |
| Roll/Pitch 最大值 | 0.89° / 2.92° | **0.86° / 1.79°** | 1.31° / 2.33° | 1.40° / 2.08° |
| Unexpected done | 0 | 0 | 0 | 0 |

相对 `model_13500`，两条平滑模型的平均 tracking 分别改善 `30.6%/31.0%`，轮 torque RMS 降低 `54.6%/50.6%`，轮 power RMS 降低 `63.8%/54.8%`。S4 起点平滑模型的腿 torque mean 降低 `6.1%`、腿 power RMS 降低 `10.9%`，但腿 torque RMS 增加 `0.7%`、横移+yaw hip RMS 增加 `1.0%`，没有达到两项降低 10% 的目标。两条平滑模型在横移中均出现过 `17 Nm` 轮扭矩上限，超过额定 90% 的样本比例最高约 `0.0104%`；虽然低于 1% 压力上限，但高于 `model_13500` 的零饱和基线，因此不满足“不新增饱和”门槛。

| 命令 | 目标 | `model_13500` 实际 / RMSE | 平滑自 `model_13500` 实际 / RMSE | 平滑自 S4 实际 / RMSE |
| --- | ---: | ---: | ---: | ---: |
| 前进 | +0.600 m/s | +0.5766 / 0.0235 | +0.5737 / 0.0263 | +0.5888 / 0.0113 |
| 后退 | -0.600 m/s | -0.5908 / 0.0093 | -0.5801 / 0.0199 | -0.5796 / 0.0227 |
| 左横移 | +0.600 m/s | +0.5634 / 0.0732 | +0.5637 / 0.0585 | +0.5520 / 0.0642 |
| 右横移 | -0.600 m/s | -0.5692 / 0.0717 | -0.5529 / 0.0652 | -0.5496 / 0.0650 |
| 左 yaw | +0.600 rad/s | +0.5626 / 0.0712 | +0.5843 / 0.0364 | +0.6073 / 0.0346 |
| 右 yaw | -0.600 rad/s | -0.5435 / 0.0736 | -0.5960 / 0.0177 | -0.5901 / 0.0250 |

平均 tracking 的提升主要来自 yaw 和横移。S4 平滑模型的后退 RMSE 相对 `model_13500` 增加约 `143.5%`，因此不满足“任何单项命令不得退化”。

| 模式 | `model_13500` 峰值 / duty | 平滑自 `model_13500` 峰值 / duty | 平滑自 S4 峰值 / duty |
| --- | ---: | ---: | ---: |
| 左横移 | 245.65 N / 0.825 | 573.41 N / 0.564 | 616.53 N / 0.570 |
| 右横移 | 262.80 N / 0.870 | 535.16 N / 0.571 | 597.57 N / 0.577 |
| 左 yaw | 246.34 N / 0.939 | 255.31 N / 0.688 | 247.44 N / 0.675 |
| 右 yaw | 223.21 N / 0.935 | 243.37 N / 0.680 | 248.72 N / 0.688 |

接触占空比从约 `0.89` 降到 `0.63`，表明接触时间显著下降，与更明显的摆动/抬脚行为一致；当前评估器没有直接测量足端 clearance，因此不能仅凭 duty 给出抬脚高度。横移落脚峰值超过 `450 N`，其中 S4 平滑模型右横移 P99 已达到 `531.19 N`，不是单个异常点。同一 seed 重复评估得到完全相同的 `616.53 N` 峰值。

为定位训练阶段，额外评估 S4 分支的 G1 `model_999.pt`。左右横移实际主轴速度为 `+0.5875/-0.5926 m/s`，接触峰值为 `272.15/286.14 N`，duty 为 `0.746/0.721`，均明显优于 G2 最终模型；G2 后左右横移 RMSE 分别增加约 `77.6%/47.4%`，峰值接触力增加约 `126.5%/108.8%`。G1 前进/后退实际速度为 `+0.5963/-0.5878 m/s`，RMSE 为 `0.00435/0.01219`；左 yaw 仍较弱，实际 `0.5487 rad/s`、RMSE `0.09399`。G1 静止漂移为 `0.003303 m/s`，腿/hip RMS 为 `7.486/2.247 Nm`，说明静止内力问题主要继承自 S4 起点；G2 将漂移改善到 `0.000599 m/s`，但没有降低静止扭矩。因此 G2 的收益集中在静止漂移和 yaw，横移速度与冲击退化则在 G2 课程中形成；具体责任项仍需单变量消融。

正式判断：整套平滑训练配方与平均 tracking、轮端负载和接触占空比改善相关，同时横移接触峰值显著恶化；本轮一次改变了命令分布、姿态/hip/mirror、动作与加速度、slip、clearance、最小接触数和随机化，不能把结果归因于单个奖励。`swing_clearance=-100` 等配置导致重落地是下一轮需要单独验证的假设。两条模型均未通过接触力、腿 RMS/hip 降幅、逐命令 tracking 和新增轮饱和门槛；其中从 `model_13500` 开始的分支还以 `0.007889 m/s` 静止漂移超过 `0.002 m/s` 门槛。两者均不晋级，继续保留 `model_13500` 作为已验证发布基线。

本次为固定 seed 单次筛查；正式发布仍需多 seed 复验。评估真源：`outputs/lateral_smooth_fixed_eval_20260902/results/`，重复验证文件为 `lateral_from_s4_7996_repeat.json`。
