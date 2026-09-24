# Reward C：从 model_13800 续训

状态：**代码与训练门控已归档，尚未在 GPU 上启动。**

2026-09-03 训练机切换到无卡模式后，CUDA/PhysX 训练与固定回放不可执行；恢复 GPU 后从本文件的父 checkpoint 和 SHA 校验继续。

## 父 checkpoint

| 字段 | 值 |
| --- | --- |
| Run | `2026-09-02_05-32-28_rough_v0_4k_stumble_n01` |
| Checkpoint | `model_13800.pt` |
| Bytes | `6,010,303` |
| SHA256 | `b79c66749f5e1147e1c780c0ba7a35790c212b28f8789012e6fb97c8097e0727` |
| Task | `DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0` |
| Seed / envs | `42 / 4,096` |

checkpoint 不进入普通 Git。启动器会在训练前校验 SHA256，并在每个 run 的 manifest 中记录 Git commit、dirty 状态、Python 版本、父文件大小与实际 SHA。

## 相对父模型的全部修改

| 配置 | 父模型 | Reward C |
| --- | ---: | ---: |
| `undesired_contacts` | `-1.0` | `0.0` |
| `contact_forces` | `-0.01` | `0.0` |
| `touchdown_velocity_l2` | `-0.2` | `0.0` |
| `feet_stumble` | `-0.1` | `-1.0` |
| `upward` | `+3.0` | `+2.0` |
| `track_lin_vel_xy_exp` | `+3.75` | `+5.0` |
| `track_ang_vel_z_exp` | `+1.875` | `+3.0` |
| 命令分布 | `10/10/10/70` | `10/10/10/70` |

其余配置继承父 run。三项移除用 `weight=0` 实现，保留 TensorBoard term 和回滚能力。

## 执行链

`queue_reward_c_from_13800.sh`：

1. 从已校验的 `model_13800.pt` 运行 200 iterations smoke。
2. 校验保存后的 YAML、命令比例、关键 scalar 数量、有限性和宽松健康阈值。
3. 从 smoke checkpoint 完整恢复，再运行 1,800 iterations。
4. 对最终 checkpoint 执行楼梯上/下和 random rough 固定评估。
5. 记录两阶段墙钟时间、秒/iteration、最终 checkpoint 与评估路径。

```bash
bash scripts/experiments/rough/queue_reward_c_from_13800.sh
```

smoke gate 只用于阻止配置错误、NaN/Inf 和明显训练崩溃，不代表效果提升。是否晋级必须比较 A/B/C 的固定评估结果。评估 JSON 会直接从传感器记录与 reward weight 无关的原始指标：非足端接触率（`>1 N`）、足端力 P95/P99/peak、touchdown 向下速度 P95/P99/peak、stumble rate（水平力 `>4x` 垂直力），另含倾角、tracking RMSE 和楼梯成功率。

其中足端力是 contact sensor 三帧 history window 内的最大**法向合力**，再跨控制步、环境和足端统计分位数；不是包含摩擦切向力的总接触力。`non_foot_contact_rate` 的分母是环境控制步（该步任一非足 body 超阈值即记一次），`stumble_rate` 同样是环境控制步（该步任一足端满足条件即记一次），两者都不是逐 body/逐足事件率。评估 JSON 的 config 会保存 history length、采样口径、阈值与 rate 分母。

## 运行产物

- RSL run：`logs/rsl_rl/doso_thunder_terrain_inward_rough/`
- 队列与 manifest：`outputs/thunder_rough_reward_c_13800/`
- 实际解析配置：每个 run 下的 `params/env.yaml`、`params/agent.yaml`

最终 checkpoint 的 bytes/SHA256 同时写入队列日志和评估 JSON。上述运行路径均由 `.gitignore` 排除；若要长期上传少量结论 JSON，应复制到 `docs/experiments/rough/results/`，保留评估 JSON 自带的 checkpoint 路径、bytes 与 SHA256，不要直接提交 `outputs/`。
