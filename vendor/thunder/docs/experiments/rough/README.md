# Thunder Rough 训练入口

本目录保存 rough terrain 的可复现配置、模型血统和实验结论。GitHub 只保存代码、参数覆盖、校验规则和小型结论文件；checkpoint、TensorBoard events、完整日志与评估产物继续保存在训练机。

## 唯一代码真源

| 内容 | 路径 |
| --- | --- |
| 共享 Flat/Rough 环境与奖励 | `source/doso_train/doso_train/tasks/terrain_inward/env_cfg.py`、`mdp.py` |
| Rough V0 增量配置 | `source/doso_train/doso_train/tasks/terrain_inward/rough_v0_env_cfg.py` |
| Gym 任务注册 | `source/doso_train/doso_train/tasks/terrain_inward/__init__.py` |
| 训练与续训入口 | `scripts/experiments/rough/` |
| 固定地形评估器 | `scripts/reinforcement_learning/rsl_rl/evaluate_rough.py` |
| 合约测试 | `tests/test_rough_v0_contract.py` |

`brain/train/rough/terrain_inward/` 和 `.doso_train_build/` 是历史工作副本，不是运行时真源，也不进入本仓库。

## 模型血统

父 run：

`logs/rsl_rl/doso_thunder_terrain_inward_rough/2026-09-02_05-32-28_rough_v0_4k_stumble_n01/model_13800.pt`

- 大小：`6,010,303 bytes`
- SHA256：`b79c66749f5e1147e1c780c0ba7a35790c212b28f8789012e6fb97c8097e0727`
- 初始化：从零训练，seed `42`，`4,096 envs`
- 命令采样：静止/纯 Yaw/纯横移/混合 = `10/10/10/70`
- 实际额外覆盖：`feet_stumble=-0.1`

截至 2026-09-03 切换无卡模式前，服务器文件显示：

| 分支 | 命令采样 | 最终 checkpoint | 结论状态 |
| --- | --- | --- | --- |
| A | `10/10/10/70` | `model_15799.pt` | 训练完成；待统一固定评估 |
| B | `5/5/5/85` | `model_15799.pt` | 训练完成；待统一固定评估 |
| Reward C | `10/10/10/70` | 尚未产生 | 启动器与门控已准备；无 GPU 时不运行 |

在线 reward 或 terrain level 只能判断训练趋势，不能证明上下楼梯、非足端接触和落脚冲击改善。A/B/C 必须使用相同 checkpoint 读取方式、seed、地形、difficulty 和动作套件做固定评估。

## 入口

- `run_from_zero_v0.sh`：复现 `model_13800` 的从零训练口径。
- `run_command_sampling_ab_from_13800.sh`：A/B 命令分布实验。
- `run_reward_c_from_13800.sh`：单阶段 Reward C 续训。
- `queue_reward_c_from_13800.sh`：200 轮健康检查、1,800 轮续训和固定评估。
- `evaluate_checkpoint.sh`：楼梯上下与 random rough 固定评估套件。

所有脚本从自身位置解析仓库根目录，也支持：

```bash
DOSO_ROOT=/path/to/doso-train \
ISAAC_PYTHON=/path/to/isaaclab/bin/python \
DEVICE=cuda:0 \
bash scripts/experiments/rough/evaluate_checkpoint.sh
```

更换父 checkpoint 时必须同时传入 `EXPECTED_SOURCE_SHA256`；默认 `model_13800.pt` 的已知 SHA 已固化在 A/B 和 Reward C 启动器中。

## 文档

- [Rough V0 从零配置](v0-from-zero.md)
- [Reward C 从 model_13800 续训](reward-c-from-13800.md)
