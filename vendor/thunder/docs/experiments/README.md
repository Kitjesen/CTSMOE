# Thunder 训练历史

本目录是 GitHub 上的训练历史入口。所有时间使用 Asia/Shanghai；训练日志、checkpoint 和评估 JSON 保存在训练机，GitHub 记录可复现参数、模型血统、结果结论及其精确路径。

## 记录规则

每个训练任务必须记录：

1. 开始/结束时间、状态和失败/重启原因。
2. Task ID、起点 checkpoint、恢复方式、环境数、seed、iterations 和训练耗时。
3. 相对上一任务修改的全部参数；未修改部分注明继承来源。
4. 最终 checkpoint、训练日志、进度日志和评估结果路径。
5. 固定评估的 tracking、扭矩、功率、接触力、姿态和终止指标。
6. 明确结论：已验证提升、未通过、停止或待验证。训练 reward 不单独作为效果结论。

详细参数和指标见 [2026-09-01 Thunder 20k、Post-20k 与侧移平滑记录](2026-09-01-thunder-torque-tracking-20k.md)。

完整参数采用双真源：GitHub 训练脚本记录相对环境默认值的精确覆盖；每个 RSL run 的 `params/env.yaml` 和 `params/agent.yaml` 记录实际解析后的完整配置。

- [20k 从零训练脚本](../../scripts/experiments/run_thunder_20k_queue.sh)
- [Post-20k 负载课程脚本](../../scripts/experiments/run_thunder_post20k_payload.sh)
- [侧移与纯 yaw 平滑脚本](../../scripts/experiments/run_thunder_lateral_recovery.sh)
- [G1-R 完整扰动与延迟鲁棒性脚本](../../scripts/experiments/run_thunder_g1r_latency.sh)
- [Rough 训练、A/B、Reward C 与固定评估](rough/README.md)

Rough 的正式 Python 实现继续与 Flat 共享 `terrain_inward` 模块；只有实验启动器和文档按 rough 子目录聚合，避免复制共享 MDP 和环境配置。

## 20k 从零训练

共同条件：Flat Thunder、`32,768 envs`、seed `42`、20 秒 episode、完整 domain randomization。

| 任务 | 时间 | 起点/训练量 | 参数变化 | 效果与状态 |
| --- | --- | --- | --- | --- |
| `fs20k_kp90_tw3e5_track125` | 2026-09-01 03:29-14:43 | 从零；20,000 iterations | `Kp/Kd=90/6.93`；腿 torque `-3e-5`；tracking `3.75/1.875` | 完成。`model_13500` 相对 `model_11994`：tracking RMSE `-27.4%`、腿 RMS `-25.8%`、横移/yaw hip RMS `-38.1%`；最终 `model_19999` 相对 `model_13500` tracking 恶化 `6.4%`、轮 RMS 增加 `11.0%`，未晋级。 |
| `fs20k_kp120_tw3e5_track125` | 2026-09-01 14:43-16:12 | 从零；计划 20,000，实际约 2,600 | 仅将腿 `Kp/Kd` 改为 `120/8` | 为 Post-20k 安全停止；最后完整 `model_2600.pt`，未做正式固定评估。 |
| `fs20k_kp60_tw3e5_track125` | 未启动 | 计划从零；20,000 | 计划 `Kp/Kd=60/5.66` | 优先级切换后取消，未产生 checkpoint。 |
| `fs20k_baseline_kp120_tw1e5_track100` | 未启动 | 计划从零；20,000 | 计划 `Kp/Kd=120/8`、腿 torque `-1e-5`、tracking `3.0/1.5` | 未产生 checkpoint。 |

运行真源：`outputs/thunder_20k_full_dr_20260901/progress.log`。

## Post-20k 负载课程

共同条件：Flat Thunder、`32,768 envs`、seed `42`、`Kp/Kd=90/6.93`；命令分布改为静止 15%、纯 yaw 15%、纯横移 10%、混合 60%。腿 torque/power 为 `-3e-5/-1e-5`，tracking 为 `3.75/1.875`，Gain DR 为 `0.8-1.2`。S1 轮 torque/power 为 `-7.5e-4/-1.5e-5`，S2-S4 为 `-1.5e-3/-3e-5`；其余逐阶段随机化范围以训练脚本和保存的 YAML 为准。

| 阶段 | 时间 | 起点/训练量 | 参数变化 | 效果与状态 |
| --- | --- | --- | --- | --- |
| S1 `s1_nominal_repair` | 2026-09-01 16:38-17:13 | `model_19999` actor/std warm-start；1,000 | 空载；Gain DR `0.8-1.2`；关闭持续外力 | 完成 `model_999`。三运动命令 tracking 较 `model_19999` 改善 `7.7%`，但静止漂移、腿扭矩和接触峰值未通过门槛。 |
| S2 `s2_payload_4kg` | 2026-09-01 17:13-18:20 | 完整恢复 S1；2,000 | Payload `0-4 kg`、50% 零载；外力/外矩 `±10 N/±3 Nm` | 完成 `model_2998`；未发布独立固定评估结论。 |
| S3 `s3_payload_8kg` | 2026-09-01 18:20-19:44 | 完整恢复 S2；2,500 | Payload `0-8 kg`、40% 零载；外力/外矩 `±20 N/±6 Nm` | 完成 `model_5497`；未晋级为默认模型。 |
| S4 `s4_full_payload` | 2026-09-01 19:44-21:11 | 完整恢复 S3；2,500 | Payload `0-12 kg`、35% 零载；外力/外矩 `±30 N/±10 Nm` | 完成 `model_7996`；固定评估相对 `model_13500` tracking 改善 `18.6%`，但腿 RMS 增加 `13.5%`、hip RMS 增加 `25.9%`、静止漂移为 `0.00501 m/s`，未晋级。 |

运行真源：`outputs/thunder_post20k_payload_20260901/progress.log` 和 `manifests/stages.ndjson`。

## 侧移与纯 yaw 平滑双基线

两条分支使用相同课程：Flat Thunder、`4,096 envs`、seed `42`、`Kp/Kd=90/6.93`；静止/纯 yaw/纯横移/混合为 `10/20/30/40%`。G1 actor/std warm-start 1,000 iterations，G2 完整恢复后追加 4,000；G2 将 `action_rate_l2` 加强到 `-0.02`、腿/轮加速度加强到 `-5e-7/-1e-9`，并提高 hip 限位、wheel slip、swing clearance 和最小接触轮数惩罚。

| 任务 | 时间 | 策略初始化 | 最终 checkpoint | 效果与状态 |
| --- | --- | --- | --- | --- |
| `lateral_smooth_g1_v2 -> g2_v2` | 2026-09-01 21:45-23:14 | `model_13500.pt` actor/std | `2026-09-01_22-02-31_lateral_smooth_g2_v2/model_4998.pt` | 完成。相对基线 tracking 改善 `30.6%`、轮 RMS 降低 `54.6%`，但腿 RMS 增加 `2.5%`、静止漂移 `0.00789 m/s`、接触峰值 `573.41 N`，未通过。旧 v1 因 wheel contact geometry 错误停止。 |
| `lateral_smooth_s4_7996_g1_v2 -> g2_v2` | 2026-09-01 23:06 至 2026-09-02 00:33 | 等效 28k 的 S4 `model_7996.pt` actor/std | `2026-09-01_23-27-27_lateral_smooth_s4_7996_g2_v2/model_4998.pt` | 完成。tracking 改善 `31.0%`、腿 mean 降低 `6.1%`、轮 RMS 降低 `50.6%`，静止漂移通过；但腿 RMS/hip 未改善且横移接触峰值 `616.53 N`，未通过。 |

阶段定位：S4 分支 G1 `model_999.pt` 的左右横移速度为 `+0.5875/-0.5926 m/s`、接触峰值为 `272.15/286.14 N`；前后跟踪也优于 G2，但静止漂移 `0.003303 m/s`、左 yaw RMSE `0.09399` 仍未通过。G2 改善静止漂移和 yaw，却造成横移速度与冲击退化。具体责任参数待单变量消融。

运行真源：

- `outputs/thunder_lateral_smooth_4096_20260901/progress.log`
- `outputs/thunder_flat_post20k_s4_7996_lateral_smooth_4096_20260901/progress.log`
- `outputs/lateral_smooth_fixed_eval_20260902/results/`

## G1-R 完整扰动与延迟鲁棒性

详细设计、硬门槛和分支血统见 [2026-09-02 Thunder G1-R 实验记录](2026-09-02-thunder-g1r-latency.md)。名义恢复和后续延迟分支使用 `4,096 envs`、seed `42`，共同起点是 S4-G1 `model_999.pt`；G2 不再用于完整运动模型，只在隔离的纯 Yaw 专项中复用其 actor/std。

| 任务 | 时间 | 起点/训练量 | 参数变化 | 效果与状态 |
| --- | --- | --- | --- | --- |
| `g1r_a` | 2026-09-02 04:01 开始；完成 5,000 | S4-G1 actor/std warm-start；恢复后扩展至 5,000 | 恢复 0–12 kg、完整 reset/外扰；命令 `10/15/15/60%`；`joint_mirror=-0.10` | 横移速度和接触合格；但 yaw 最低 `0.523 rad/s`、横移姿态 RMS `1.31°`、静止漂移 `0.0206 m/s`、静止 hip RMS `5.91 Nm`，未通过。 |
| `g1r_b` | 2026-09-02 完成 2,000 后停止 | 同一 S4-G1 actor/std | mirror `-0.15`，但 yaw scale `0.5`、横移 scale `0.0` | 与全模式对称目标冲突且右 yaw 仅 `-0.4983 rad/s`，作废，不再续训。 |
| `g1r_symmetric_recovery_8k` | 2026-09-02 10:38-11:47 | G1R-A 5k 完整恢复；运行至 iteration 5,981/7,997 | 所有模式 mirror `-0.15`；姿态、静止与 yaw tracking 加强 | 为切换纯 Yaw 特训而停止并保留 checkpoint；最后在线 mean reward `307.11`，尚无固定评估。 |
| `LAT-OBS/DROP/ACT/COMBINED` | 名义选模后 | 名义优胜 actor/std；合计 3,500 | 0–3 帧观测延迟、丢帧保持和动作 FIFO，最后组合课程 | 代码与 CPU/Kit 契约已验证；GPU rollout 及训练待名义选模。 |
| `yaw_specialization_g2_3k` | 2026-09-02 11:47-14:54 | S4-G2 `model_4998.pt` actor/std；3,000 | 纯 yaw 50%；全模式 mirror `-0.20`；轮速 mirror `-0.002`；yaw tracking `2.5/std 0.25`；姿态、4 cm clearance 与 touchdown 收紧 | **固定评估失败**：空载Yaw仅 `+0.547/-0.554`，左Yaw峰值 `511.6 N`，静止漂移约 `0.008 m/s`；12kg峰值 `606.1 N`。禁止部署和继续作为默认父模型。 |
| `yaw_soft_touchdown_2k` | 2026-09-02 14:58-16:06 | 失败的Yaw `model_2999.pt` actor/std；计划2,000，实际运行至1,151 | 保持4 cm clearance；touchdown `-0.8`、速度阈值 `0.10 m/s`；接触软目标 `300 N` | **因父模型固定评估失败停止**；最后完整 `model_1100.pt`，仅保留为失败实验，不训练、不部署。 |

运行真源：`outputs/thunder_g1r_latency_20260902/progress.log`、`manifests/stages.ndjson` 和 `results/`。

## Run 与 artifact 对照

下列均为训练机工作区 `/root/autodl-tmp/workspace/doso-train/` 下的相对路径。每个 run 目录内的 `params/env.yaml`、`params/agent.yaml` 是完整配置快照。

| 任务 | RSL run / checkpoint | 训练日志与结果 |
| --- | --- | --- |
| 20k Kp90 | `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_03-29-35_fs20k_kp90_tw3e5_track125/model_19999.pt` | `outputs/thunder_20k_full_dr_20260901/logs/train_fs20k_kp90_tw3e5_track125.log`；`outputs/thunder_20k_full_dr_20260901/results/fs20k_kp90_tw3e5_track125.json` |
| 20k Kp120 | `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_14-43-08_fs20k_kp120_tw3e5_track125/model_2600.pt` | `outputs/thunder_20k_full_dr_20260901/logs/train_fs20k_kp120_tw3e5_track125.log`；无正式评估 |
| Post-20k S1 | `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_16-38-49_s1_nominal_repair/model_999.pt` | `outputs/thunder_post20k_payload_20260901/logs/train_s1_nominal_repair.log`；`results/` 内 S1 筛查 |
| Post-20k S2 | `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_17-14-05_s2_payload_4kg/model_2998.pt` | `outputs/thunder_post20k_payload_20260901/logs/train_s2_payload_4kg.log`；`results/` 内 S2 筛查 |
| Post-20k S3 | `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_18-20-57_s3_payload_8kg/model_5497.pt` | `outputs/thunder_post20k_payload_20260901/logs/train_s3_payload_8kg.log`；`results/` 内 S3 筛查 |
| Post-20k S4 | `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_19-44-59_s4_full_payload/model_7996.pt` | `outputs/thunder_post20k_payload_20260901/logs/train_s4_full_payload.log`；`results/` 内 S4 筛查 |
| 平滑 model_13500 G1/G2 | `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_21-45-20_lateral_smooth_g1_v2/model_999.pt` -> `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_22-02-31_lateral_smooth_g2_v2/model_4998.pt` | `outputs/thunder_lateral_smooth_4096_20260901/logs/train_lateral_smooth_g1_v2.log`、`outputs/thunder_lateral_smooth_4096_20260901/logs/train_lateral_smooth_g2_v2.log` |
| 平滑等效 model_27999 G1/G2 | `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_23-06-48_lateral_smooth_s4_7996_g1_v2/model_999.pt` -> `logs/rsl_rl/doso_thunder_terrain_inward_flat/2026-09-01_23-27-27_lateral_smooth_s4_7996_g2_v2/model_4998.pt` | `outputs/thunder_flat_post20k_s4_7996_lateral_smooth_4096_20260901/logs/train_lateral_smooth_s4_7996_g1_v2.log`、`outputs/thunder_flat_post20k_s4_7996_lateral_smooth_4096_20260901/logs/train_lateral_smooth_s4_7996_g2_v2.log` |

## 当前结论

- 当前经过固定评估、可作为已验证基线的是 `model_13500`。
- `model_19999` 未带来净提升，主要问题是 yaw/静止退化以及轮 torque/power 上升。
- Post-20k `model_7996` 和两条侧移平滑最终模型均已完成同口径固定评估。整套训练配方改善了平均 tracking、轮端负载和接触占空比，但没有通过腿 RMS/hip、逐命令 tracking、轮饱和与 `450 N` 接触力门槛，不替换 `model_13500`。本次为固定 seed 筛查，正式发布仍需多 seed 复验。
- 血统准入规则：失败父模型的后代只能作为研究分支；后代在独立完整固定评估通过前不得部署，也不得继续作为默认训练父模型。`yaw_specialization/model_2999` 及其 `yaw_soft_touchdown` 后代已按此规则作废。
