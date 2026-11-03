# Direct 与 settle 前进命令切入附录

## 结论

V12 direct 前进的 **1.0284rad** 最大单关节偏移来自 `fl_calf_joint`，发生在 **0.58s**，并非此前重点查看的 `rl_thigh_joint`。该时刻原请求为 -1.0161rad，guard 后、actuator delay 前的 `data.joint_pos_target` 为 -1.0161rad，actual 为 -1.0284rad，两者差 -0.0123rad。这个峰与 guard 后 target 贴近，支持“策略请求把受限目标持续推到 calf 下限附近”；但收据未保存 delay buffer 实际送入 PD 的位置目标，所以不能把该差值严格解释成执行器跟踪误差，也不能仅凭本 trace 完全排除 delay 层内的瞬态。

最早时序也支持这个区分：direct 的 request 首次超过 0.60rad 在 **0.10s**（`fr_thigh_joint`），guard 后 target 首次超过 0.60rad 同样在 **0.10s**，actual qRMS 在 **0.14s** 越过 0.25rad，单关节 actual 到 **0.18s** 才超过 0.60rad。0.02s 的第一个控制步已同时观察到非默认 request/target、120Nm 转矩峰和 15rad/s 速度峰；50Hz post-step request 与控制步内 200Hz 峰无法继续判断这 20ms 内谁先发生。后续可分辨时序是 request/guarded target 先越门、actual 后越门，因此证据优先指向策略请求与受限目标序列；对未记录 delay 后目标的执行器瞬态保持保留。

settle 并不表示网络原请求始终很小：命令切入前 0.20s，`rl_hip_joint` request 相对默认姿态已约 **-4.7740rad**；切入第一步仍为 **-4.6284rad**，但 guard 后 target 保持约 **-0.1690rad**，actual 约 **-0.0159rad**。这项大 request 在零命令历史中已经存在，并非 0.6m/s 边沿新产生。settle 首秒最大 actual 偏移是 `fr_calf_joint` 的 **0.3072rad@2.44s**。所以 request 绝对幅值或 mean policy action 都不能单独解释 direct 拒门；关键观测是 direct 的多关节请求经过 guard 后仍形成了能把 actual 推过姿态门的目标序列。

## 命令边沿关键步

|模式|相对切入|qRMS|mean abs policy action|最大 request|最大 guard 后 target|最大 actual|腿转矩/速度峰|最大轮接触|
|---|---:|---:|---:|---|---|---|---:|---:|
|direct|0.00s|0.0897|0.9780|fl_hip -0.5927|fr_thigh +0.2400|fr_calf -0.1439|120.0Nm / 15.0000rad/s|0.0N|
|direct|+0.10s|0.1985|1.8084|fl_hip -0.7694|fr_thigh +0.6416|fr_thigh +0.3760|32.1Nm / 6.8124rad/s|200.5N|
|direct|+0.20s|0.4295|1.7677|fl_calf -1.1308|fl_calf -1.0230|fl_calf -0.7956|30.8Nm / 4.4025rad/s|231.2N|
|settle|-0.20s|0.0512|2.7065|rl_hip -4.7740|fl_hip +0.1690|fr_calf -0.1134|21.3Nm / 0.2677rad/s|123.4N|
|settle|0.00s|0.0519|2.9268|rl_hip -4.6284|fl_thigh -0.2731|fr_calf -0.1178|35.3Nm / 0.8174rad/s|228.2N|
|settle|+0.20s|0.0907|1.9328|rl_hip -2.5555|fl_hip +0.1690|fr_calf -0.2144|21.6Nm / 1.3071rad/s|129.6N|

direct 在 +0.20s 已达到 qRMS 0.4295，而 settle 在相同相对时刻仍为 0.0907。反过来，settle 的 mean abs action 与最大原请求更大，却被 guard 隔离在较小 target/actual 范围内。这再次说明不能按 raw action 或 request 单值排序风险，必须看 request→guarded target→actual 的整条时序。

## 两个 case 的真实差别

- evaluator 对两者使用同一 V12 student checkpoint、同一 seed、同一 flat 几何、同一默认关节/根状态、关闭 observation corruption 和其它 events。两者 case 定义只在 `mode` 不同。
- direct 从 reset 后第一帧就给 `vx=0.6m/s`；settle 在前 2 秒给零命令，随后一步切到相同 `vx=0.6m/s`。CSV 中 settle 有命令切入前 0.2 秒数据；direct 的命令在首个 policy observation 前已经非零，因此不存在同一 episode 内的负时间控制步，只有 manifest 的初始状态可作 t=0 参考。
- policy observation 是 10 帧历史，每帧 53 维。direct 使用 reset 起点的历史立即动作；settle 在命令切入时已有约 100 个站立控制步，输入的最近 10 帧覆盖约 0.2 秒零命令站立状态。这是两种协议除命令时刻外最重要的输入历史差异。
- evaluator 实例化的是 `Go2Env`，不是带随机子步 action delay 的 `ActionDelayGo2Env`。所有 actuator 的 delay 固定为 2 个 5ms physics steps，即 10ms；direct 与 settle 相同。
- 训练的 `omni_full_episode_v1` 保留 `normal_reset_root/joints` 和普通 5 秒 command sampler；但 evaluator 在配置 profile 后覆盖 reset 函数，强制默认关节、零速度和零 pose/velocity range，再用脚本命令。这里的 direct/settle 是 matched fixed-case 诊断，settle 不是训练 sampler 的独立随机分支。

## 控制步文件

- `onset_control_steps.csv`：两 case 命令切入前后 0.2 秒；主要关节的 policy raw action、原请求、guard 后/delay 前 target、actual、误差、关节速度、控制步内 200Hz 转矩/速度峰，以及全动作均值、base 与四轮接触。
- `onset_key_steps.csv`：关键相对时刻的宽表，直接给出全腿最大 request/target/actual、qRMS、动作均值、转矩、速度和轮接触。
- `onset_summary.json`：门限首次越界、最大单关节快照、输入 case 与源码契约。

注意：trace 中的 `target` 是 `robot.data.joint_pos_target`。本次实际 remote runtime 的 `/root/autodl-tmp/workspace/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py` 已核对：第 1101 行写 `data.joint_pos_target`，第 1880 行把该字段组装进 control action，第 1886 行才调用 `actuator.compute()`，第 1893 行仅在返回位置非空时写独立 sim target。线上同版本 [`articulation.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/assets/articulation/articulation.py#L1490-L1497) 也把 data target 定义为“sent to the actuator from the user”。[`DelayedPDActuator.compute`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/actuators/actuator_pd.py#L323-L331) 先用 `positions_delay_buffer.compute()` 产生送入 PD 的延迟位置目标，而显式 [`IdealPDActuator.compute`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/actuators/actuator_pd.py#L167-L181) 随后输出 effort 并清空 `joint_positions`。因此对本次 Thunder 显式电机，`_joint_pos_target_sim` 不是可预设为有效的 delay 后位置目标；本收据能区分 original request、guard 后/delay 前 target 和 actual，**没有**记录 delay buffer 实际送入 PD 的位置目标。

## 下一单因素待验假设

保持初始状态、策略、固定 10ms delay 和 `vx=0.6m/s` 不变，只改变命令边沿，比较 t=0 step 与 0.2/0.5/1.0 秒线性 ramp；同时新增 guard 前 request、guard 后/delay 前 target，以及 `positions_delay_buffer.compute()` 返回并送入 PD 的位置目标，配套记录 computed/applied effort。另可在**离线网络输入反事实**中替换历史帧，判断 actor 输出对历史的敏感性，但复制 settle 历史会脱离 reset 时真实物理状态，不能作为 on-policy 行为能力或 gate 验收。该建议是待验假设，不是现有数据的因果结论。
