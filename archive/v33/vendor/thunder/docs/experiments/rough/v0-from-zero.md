# Thunder Rough V0 从零训练冻结配置

状态：**配置与实际训练血统已归档。实际 4,096-env 运行保留到 `model_13800.pt`。**

日期：2026-09-02（Asia/Shanghai）

Task：`DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0`

实现：`source/doso_train/doso_train/tasks/terrain_inward/rough_v0_env_cfg.py`

启动脚本：`scripts/experiments/rough/run_from_zero_v0.sh`

## 1. 训练血统与冻结原则

- 从零初始化 actor、critic、action std 和 optimizer。
- 不使用 `--resume`、`--checkpoint`、`--actor-warm-start`。
- 训练上限 `20,000 iterations`，每 `100 iterations` 保存一次。
- seed `42`；配置默认 `32,768 envs`，实际父模型运行使用 `4,096 envs`；每个环境每轮采样 `24 policy steps`。
- 奖励、动作、命令、reset 和域随机化以已验证的 flat `model_13500` 保存配置为基线。
- V0 只额外引入 rough terrain、critic 特权信息、轮力矩/轮功率惩罚；包内默认 `feet_stumble=0`。实际 `model_13800` 血统由启动脚本单独覆盖为 `-0.1`。
- 本文件和每个 run 的 `params/env.yaml`、`params/agent.yaml` 共同构成配置真源。

## 2. 时间尺度与 episode

| 项目 | 值 |
| --- | ---: |
| Physics dt | `0.005 s` / `200 Hz` |
| Decimation | `4` |
| Policy period | `0.020 s` / `50 Hz` |
| Episode | `20 s` / `1000 policy steps` |
| Steps per env per PPO iteration | `24` |
| Environments | 配置默认 `32,768`；实际父模型 `4,096` |
| Iterations | `20,000` |

## 3. Actor 观测：单帧 53 维

Actor 保持实机可获得的单帧本体观测，不加入 height scan 或任何仿真特权量。

| 顺序 | 观测 | 维度 | Scale | 训练噪声 |
| ---: | --- | ---: | ---: | ---: |
| 1 | Base angular velocity | 3 | `0.25` | `±0.2` |
| 2 | Projected gravity | 3 | `1.0` | `±0.05` |
| 3 | Velocity command | 3 | `1.0` | 无 |
| 4 | 12 个腿关节相对位置 | 12 | `1.0` | `±0.01` |
| 5 | 16 个关节相对速度 | 16 | `0.05` | `±1.5` |
| 6 | 上一动作 | 16 | `1.0` | 无 |

合计：`3 + 3 + 3 + 12 + 16 + 16 = 53`。

不包含 base linear velocity、四个轮位置、历史帧、Kp/Kd、摩擦和 height scan。

## 4. Critic 观测：274 维

### 4.1 原始状态与地形

| 观测 | 维度 |
| --- | ---: |
| Base linear velocity | 3 |
| Base angular velocity | 3 |
| Projected gravity | 3 |
| Command | 3 |
| 腿位置 | 12 |
| 全关节速度 | 16 |
| 上一动作 | 16 |
| 小计 | `56` |
| Height scan | `187` |

Height scan：范围 `1.6 × 1.0 m`、分辨率 `0.1 m`、`17 × 11 = 187`，只提供给 critic。

### 4.2 动力学特权量

| 观测 | 维度 | 编码 |
| --- | ---: | --- |
| 腿 Kp | 12 | 实际 stiffness / nominal stiffness；顺序为 hip 4、thigh 4、calf 4 |
| 全关节 Kd | 16 | 实际 damping / nominal damping；顺序为 hip 4、thigh 4、calf 4、wheel 4 |
| 足端材质 | 3 | 实际 static friction、dynamic friction、restitution |

最终：`56 + 187 + 12 + 16 + 3 = 274`。

使用比例而不是原始 Kp/Kd 数值，是为了让输入保持在约 `0.5～2.0`，避免 `90` 与 `6.93` 的量纲差异直接进入未归一化 critic。critic 无观测噪声；actor 维度不变，因此实机 ONNX 输入契约仍为 53 维。

当前未加入的候选特权量：base mass、base COM、其他刚体质量、payload 质量/位置、motor torque scale、时延状态。V0 不随机 payload、torque scale 或时延；质量/COM 已随机但暂不为此增加新的缓存与观察通路。若 V0 证明 privileged critic 有价值，再单独做质量/COM 消融。

## 5. 动作与执行器

动作总计 16 维：

- 12 个腿位置动作，FR → FL → RR → RL，每条腿 hip/thigh/calf。
- Hip scale `0.125 rad`；thigh/calf scale `0.25 rad`；使用 default pose offset。
- 4 个轮速度动作，FR → FL → RR → RL，scale `5 rad/s`。
- 动作 clip `[-100, 100]`。

名义执行器：

| 执行器 | Effort | Velocity | Kp | Kd |
| --- | ---: | ---: | ---: | ---: |
| Hip/Thigh/Calf | `120 Nm` | `17.48 rad/s` | `90` | `6.93` |
| Wheel | `17 Nm` | `44 rad/s` | `0` | `1` |

每次 reset 对所有关节的 stiffness 与 damping 独立乘 `0.5～2.0`；critic 看到实际比例，actor 看不到。

## 6. Rough terrain

- Terrain generator：Isaac Lab `ROUGH_TERRAINS_CFG`。
- 单块尺寸 `8 × 8 m`，`10` 个 level rows，`20` 个 terrain columns。
- Horizontal scale `0.1 m`，vertical scale `0.005 m`，slope threshold `0.75`。
- 初始 level 在 `0～5` 中采样。
- Terrain static/dynamic friction 均为 `1.0`，combine mode 为 `multiply`。

地形组成：

| 地形 | 比例 | 参数 |
| --- | ---: | --- |
| 上楼梯 | 20% | step height `0.05～0.23 m`，width `0.3 m` |
| 下楼梯 | 20% | step height `0.05～0.23 m`，width `0.3 m` |
| 随机方块 | 20% | height `0.05～0.20 m`，grid width `0.45 m` |
| 随机粗糙面 | 20% | noise `0.02～0.10 m`，step `0.02 m` |
| 正坡 | 10% | slope `0～0.4` |
| 反坡 | 10% | slope `0～0.4` |

自动 curriculum：

- 行走距离超过半块地形（`4 m`）时 level +1。
- 行走距离小于命令速度对应目标距离的一半时 level -1。
- 已达到最高 level 后由 terrain importer 按其标准逻辑重新分配。

## 7. 初始姿态与 reset

名义 base：位置 `(0, 0, 0.46 m)`、四元数 wxyz `(1, 0, 0, 0)`、线速度和角速度全 `0`。

名义关节：

| 腿 | Hip | Thigh | Calf |
| --- | ---: | ---: | ---: |
| FR | `-0.20` | `-0.93` | `+1.96` |
| FL | `+0.20` | `+0.93` | `-1.96` |
| RR | `+0.20` | `+0.93` | `-1.96` |
| RL | `-0.20` | `-0.93` | `+1.96` |

四个 wheel joint position 为 `0`，全部关节速度为 `0`。

每回合 reset：

- 关节 position scale 固定 `[1.0, 1.0]`，velocity `[0, 0]`；没有 hip 专项 reset。
- Base x/y `±0.5 m`，z offset `0～0.2 m`。
- Roll/pitch/yaw 均为 `±3.14 rad`，即全姿态随机 reset。
- 线速度 xyz 均为 `±0.5 m/s`；角速度 roll/pitch/yaw 均为 `±0.5 rad/s`。
- Reset 外力 `±30 N`，外矩 `±10 Nm`。
- 每 `10～15 s` 施加 x/y 速度 push `±0.5 m/s`。

## 8. 域随机化

Startup：

- 所有机器人刚体材质：static friction `0.3～1.0`、dynamic friction `0.3～0.8`、restitution `0～0.5`、64 buckets。
- 足端用同一分布再采样一次并保存最终真实值，供 critic 读取；actor 不读取。
- Base mass：名义质量加 `-1～+3 kg`，重算 inertia。
- 其他刚体质量：名义质量乘 `0.7～1.3`，重算 inertia。
- Base COM xyz：各 `±0.05 m`。
- 无 payload、无 supply-voltage、无观测/动作延迟随机化。

Reset：

- Kp/Kd scale `0.5～2.0`。
- 完整 root pose/velocity reset、外力/外矩和 interval push，范围见上一节。

## 9. 命令分布

- 每 `10 s` 重采样。
- `lin_vel_x/y = [-1, 1] m/s`。
- `ang_vel_z = [-1, 1] rad/s`。
- heading `[-π, π]`，heading control stiffness `0.5`。
- 静止 10%，纯 yaw 10%，纯横移 10%，其余混合 70%。
- 纯命令最小幅值 `0.2`。

## 10. 奖励

V0 有 20 个非零奖励；另注册 `feet_stumble`，但权重为 `0`。

| Reward | Weight | 范围/参数 |
| --- | ---: | --- |
| `track_lin_vel_xy_exp` | `+3.75` | `std=0.5` |
| `track_ang_vel_z_exp` | `+1.875` | `std=0.5` |
| `upward` | `+3.0` | 保持原公式 |
| `feet_contact_without_cmd` | `+0.1` | 四个 foot links |
| `lin_vel_z_l2` | `-2.0` | 原公式 |
| `ang_vel_xy_l2` | `-0.05` | 原公式 |
| `joint_torques_l2` | `-3e-5` | 腿 12 关节 |
| `joint_acc_l2` | `-1e-7` | 腿 12 关节 |
| `joint_acc_wheel_l2` | `-2.5e-10` | 四轮 |
| `joint_pos_limits` | `-5.0` | 腿 12 关节 |
| `joint_power` | `-1e-5` | 腿 12 关节 |
| `wheel_torques_l2` | `-1.5e-3` | 四轮；本 V0 新启用 |
| `wheel_power` | `-3e-5` | 四轮；本 V0 新启用 |
| `stand_still` | `-2.0` | command threshold `0.1`，腿 |
| `joint_pos_penalty` | `-1.0` | stand scale `5`、velocity threshold `0.5`、command threshold `0.1` |
| `joint_mirror` | `-0.05` | FR↔RL、FL↔RR |
| `action_rate_l2` | `-0.01` | 原值 |
| `undesired_contacts` | `-1.0` | 非 foot links，threshold `1 N` |
| `contact_forces` | `-0.01` | foot links，threshold `450 N` |
| `touchdown_velocity_l2` | `-0.2` | foot links，velocity threshold `0.15 m/s` |
| `feet_stumble` | `0.0` | 候选项，当前关闭 |

明确不启用：`flat_orientation_l2`、hip 限位、wheel slip、swing clearance、minimum contacts、standing/pure-yaw 专项速度奖励。

### 为什么 `feet_stumble` 暂不开启

定义：任一足端 `||F_xy|| > 4 × |F_z|` 时记一次 stumble，并乘 upright scale。它能识别脚撞墙，但 rough 的楼梯立面、箱体边缘也会产生相同信号。RobotLab 的 Thunder rough 与 B2W rough 都将该项设为 `0`；只有部分其他轮足任务使用 `-0.1`。

因此 V0 先保留统计能力但不参与优化。若 rollout 显示大量“撞立面后卡住”且不是正常跨障，再单独做 `0` 对 `-0.05/-0.1` 消融。

## 11. Termination

- `time_out`：20 秒。
- `terrain_out_of_bounds`：buffer `3 m`，按 timeout 处理。
- 不设置 fall、base contact、bad orientation 或 low-height 提前终止，保留摔倒后恢复学习机会。

## 12. PPO 与网络

- Actor/Critic MLP：`512 → 256 → 128`，ELU。
- Initial action noise std `1.0`。
- Actor/Critic observation normalization 均关闭。
- PPO epochs `5`，mini-batches `4`，learning rate `1e-3` adaptive。
- `gamma=0.99`，`lambda=0.95`，entropy `0.01`，desired KL `0.01`。
- Clip `0.2`，value loss coefficient `1.0`，clipped value loss 开启，max grad norm `1.0`。
- 无 RND、无 symmetry config。

## 13. 启动与停止条件

复现实验运行：

```bash
bash scripts/experiments/rough/run_from_zero_v0.sh
```

脚本没有任何 checkpoint 参数，从随机 actor/critic 开始。默认值复现实际父模型口径：`4,096 envs`、`feet_stumble=-0.1`。如需只验证包内冻结默认配置，显式设置 `NUM_ENVS=32768 FEET_STUMBLE_WEIGHT=0`。

训练后必须核对保存的 YAML：actor `53` 维、critic `274` 维、level `0～5`、Kp/Kd `90/6.93`、gain DR `0.5～2.0`、两个 wheel reward 非零、实际父模型 `feet_stumble=-0.1`，且 `resume=false`。
