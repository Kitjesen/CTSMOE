# 2026-11-03 迁移验证

本轮完成训练链路迁移和隔离短测试，未启动 20000 轮长训练，也未改动旧 19538 正式检查点。

## 已验证

运行环境为 Linux / RTX 5090，Isaac Lab 3.0.0、Torch 2.11.0+cu130、Newton 1.6.1、Warp 1.17.0、Go2 自带 RSL-RL 3.3.0。上游固定版本见 `PROVENANCE.json`。

| 检查 | 结果 |
| --- | --- |
| 机器人资产、单帧/教师输入 | 9 项通过；16 关节、21 STL、限位/轴/惯量、53/309 输入切片与顺序 |
| 环境、地形、奖励、重置、候选高度参考 | 33 项通过，含逐物理样本接触计时的 5 项回归 |
| CLI 和真正的上游 CTS CPU 更新/恢复/导出 | 5 项通过 |
| 实际 Lab 配置序列化及非法值拒绝 | 2 项通过 |
| 原生新任务，32 环境 × 1002 控制步 | 接口、动作限幅、历史、重置通过；历史最大误差 0；无连续单步重置循环 |
| 原生随机初始化短训练 | 0→2 更新，exit 0；teacher encoder、actor、critic、student MoE 均有权重变化 |
| 原生检查点恢复 | 2→3 更新，exit 0；同一 run ID、配置和地形等级；控制步计数 48→72 |
| 官方回放与导出 | 200 帧/4 秒/50 Hz/1920×1080，完整解码及 0.5/1.5/3 秒抽帧检查；JIT 支持 batch 3，ONNX 校验通过，与 JIT 最大差约 3.73e-8 |

最终 **49 项不同的 CPU 测试同进程全部通过**。原生 2 更新后两个 Adam 的 step 均为 40，恢复再更新一次后均为 60；模型/优化器张量及损失均为有限数值。回合在恢复时重置，不声称中途物理状态逐位恢复。

接触计时修复前后各做了一组独立的 0→2→3 短测，共 **6 次原生诊断更新**，不重复统计 CPU 合成环境测试。最终任务课程标识为 `thunder_fresh_10level_v2`，旧组不作为同一次训练继续。最终 `model_3.pt` SHA-256：`a79f0534a2011bf1170d9ee6cffae6075f9f6ca9b742485cd44b509bb0b43f99d`。

TensorBoard 的 step 0 是上游 logger 复制第一轮指标画出的初始点，不是额外参数更新。进度以 `completed_updates` 和优化器状态核对。

## 未通过的站立检查

1002 步原生探针使用零动作和默认 PD 目标，不是已学会运动的策略：

- 32 个环境全部从 L0 开始；没有环境自然达到 20 秒超时。
- 计时修复前记录 122 次腿部接触终止，平地环境也有接触；修复后为 93 次，仍未通过站立验收。两次重置时刻不同，不能把次数变化当作策略或实际碰撞改善。
- 最后对 32 个环境显式注入超时计数，用于验证重置路径；这 32 次不能算作站立成功。
- 最终控制频率采样的轮法向力峰值约 660 N、腿部约 108 N；不能据此排除更高频物理步冲击。
- 已排除动作检查残留饱和目标、以及 fresh 裁剪改变默认关节角度；承重下沉与接触还需要后续专门检查。

因此接口测试通过不等于站立验收通过。新策略只接受了 3 次隔离更新，不具备已经验证的侧移或楼梯能力。随机策略的速度误差、初始课程等级与短回合指标不作为最终运动性能结论。

## 已修复的迁移故障

- 原入口强制恢复旧检查点；新增独立 fresh 任务和显式 native resume。
- Lab 配置含 `slice` 等类型，初次训练在构造 runner 时序列化失败；已加入明确类型编码和实际配置测试。该失败没有产生 PPO 更新。
- 导出单测在梯度模式下使用 rollout 的 inference tensor；修正测试调用模式，未因此改动策略计算。
- 旧持续接触统计把窗口中一次短碰计成整段 20 ms；新任务改为 40 个连续 2.5 ms 样本才达到 100 ms，阈值不变，500 N 峰值仍立即触发。旧 122 次均由粗粒度时长触发，但历史证据只保留窗口峰值，不能认定它们全部是假碰撞。
- 保留旧环境的同时，修正 fresh 的起始课程、动作边界及奖励语义，详见 `thunder_lab3/MIGRATION.md`。

## 复测

在已安装固定 Lab 3 / Go2 环境的 Python 中，从本项目目录执行：

```bash
python -m unittest runtime.test_robot_asset task_port.test_robot_inputs
python -m unittest compat.test_fresh_runner scripts.test_launch
python -m unittest compat.test_training_contract
python -c "from isaacsim.asset.importer.urdf import URDFImporter; import unittest; r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(['thunder_lab3.test_env_contract','height_curriculum.test_height_curriculum'])); raise SystemExit(not r.wasSuccessful())"
```

原生探针和训练命令见 `README.md`。本地已核对 40 个 Python 文件与实际 Linux 测试源码哈希一致，见 `SOURCE_MANIFEST.json`；初次导入清单保存在 `BASELINE_MANIFEST.json`。证据包 SHA-256：`b61c6682e0cdd47be0f29eb6265e12803e7fc0a9493bdce8b3c8700260f64c0b`。

完整日志、轨迹、模型和视频保留在隔离实验目录，不提交 Git。4 秒录像只证明新模型可加载回放，不能评价成熟步态。实机机械限位、站姿承重、L6/L9 通行、域随机化、MuJoCo 部署和多 GPU 未通过本轮验收。
