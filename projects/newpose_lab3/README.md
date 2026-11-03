# Thunder NEWPOSE：Lab 3 从零重训

2026-11-03 决定：继续使用 Thunder 四轮腿机器人，复用 Go2 `lab3.0` 的 CTS+MoE 框架，在 Isaac Lab 3 / Newton 中建立独立实验，目标为 **0 → 20000 次参数更新**。旧策略仅作对照，不继承其权重、优化器或地形课程进度。

**当前提供独立的从零训练入口，长训练尚未启动。** 新任务为 `Thunder-Newpose-Newton3-Fresh-v0`，不提供检查点就从随机网络和空优化器开始；恢复必须显式使用 `--resume --checkpoint`，且只接受同任务的新检查点。旧 19538 只能通过 `play --legacy` 回放。

本轮迁移差异见 [环境、地形和奖励说明](thunder_lab3/MIGRATION.md)，验证结果见 [VALIDATION.md](VALIDATION.md)。短测试的参数更新与旧正式进度、未来长训练分别记录；接口通过不代表站立或步态通过。

## 代码与资产

| 路径 | 内容 |
| --- | --- |
| `thunder_lab3/` | 八类地形、L0–L9、命令、奖励与任务注册 |
| `task_port/`、`runtime/` | Thunder 模型导入、16 个动作、观测与默认站姿 |
| `teacher/` | 教师接触/动力学输入及镜像审计辅助代码 |
| `height_curriculum/` | 相对地面高度与渐进方块候选 |
| `compat/` | 官方 CTS+MoE runner 的计数/超时适配、新模型保存恢复、导出和旧模型回放 |
| `scripts/` | 从零训练/显式恢复/回放统一入口，以及不更新策略的原生环境探针 |
| `legacy/thunder/` | `thunder_v4.urdf` 和 21 个 STL 网格 |
| `legacy/diagnostics/` | 旧奖励配置测试所需的 `env.yaml`，不是新训练配置 |
| `upstream/go2_rl_lab/` | 固定提交的上游 Git 子模块 |

## 先核对机器人

资产为 `legacy/thunder/source/doso_train/doso_train/data/Robots/DOSO/thunder/urdf/thunder_v4.urdf`。

- 12 个有限位的腿关节，使用位置目标；4 个连续轮关节，使用角速度目标。
- 四髋当前限位 ±0.41 rad。前小腿 FR 为 `[0, 3.14]`、FL 为 `[-3.14, 0]`；后小腿 RR/RL 均为 `[-3.14, 3.14]`。这些是文件配置，不是已确认的实机范围。
- 当前默认角度仍为最初 NEWPOSE，见 `runtime/thunder_contract.py`；新训练站姿未定。保留四腿弯曲方向一致的要求，按实际几何核对，不要求关节数值机械地相同。
- 质量惯量、左右结构与轴向、碰撞体、机械限位、承重稳定性仍需核对；先确认这些，再确定控制与奖励。

## 来源与环境

- Go2：`https://github.com/wty-yy/go2_rl_lab`，固定提交 `6c95d9feef4814c25eb6536e61bb38e1d1d718d4`。
- Isaac Lab：`v3.0.0`，固定提交 `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`，Newton / MJWarp。
- 现有适配基线使用 Python 3.12、Go2 自带的 RSL-RL 3.3.0；不能直接用通用 RSL-RL 替换该 CTS+MoE 实现。
- 仓库包含源码和 Thunder 网格，不包含 GPU 运行环境、模型权重、视频、轨迹、生成的 USD 或服务器连接凭据。

拉取本任务分支及固定的上游代码：

```bash
git clone --recurse-submodules --branch newpose-lab3-from-scratch https://github.com/Kitjesen/CTSMOE.git
cd CTSMOE/projects/newpose_lab3
```

若需要上游 Go2 的 LFS 资产，在安装 Git LFS 后于 `upstream/go2_rl_lab` 执行 `git lfs pull`。Thunder 的 URDF/STL 直接随本仓库保存。

保留上游自身许可证：`upstream/go2_rl_lab/LICENSE` 和 `upstream/go2_rl_lab/source/rsl_rl/LICENSE`。Thunder 资产不因此被重新授权为 MIT 或其他上游许可证。

## 运行入口

以下命令从本目录执行，需先准备上述 Linux 原生运行环境。默认只打印执行计划，增加 `--execute` 才运行。

```bash
# 随机初始化的两次更新测试，不载入旧模型。
python scripts/launch.py train \
  --urdf legacy/thunder/source/doso_train/doso_train/data/Robots/DOSO/thunder/urdf/thunder_v4.urdf \
  --num-envs 32 --updates 2 --seed 42 --run-name migration_smoke

# 明确恢复新任务的检查点，updates 表示额外更新数。
python scripts/launch.py train \
  --urdf legacy/thunder/source/doso_train/doso_train/data/Robots/DOSO/thunder/urdf/thunder_v4.urdf \
  --num-envs 32 --updates 1 --seed 42 --resume --checkpoint /absolute/path/model_2.pt

# 新任务的动作、输入、接触及超时探针，不更新策略。
python scripts/probe_native.py --fresh \
  --urdf legacy/thunder/source/doso_train/doso_train/data/Robots/DOSO/thunder/urdf/thunder_v4.urdf \
  --output results/native_fresh --num-envs 32 --steps 1002 --visualizer none --device cuda:0
```

原生恢复核对模型、两个 Adam 优化器、实际配置、课程和随机数状态；回合重新开始，不保证物理轨迹逐位连续。`model_N.pt` 中 N 为已完成参数更新数，累计上限 20000。

新配置从 L0 开始，保持 10 级、8 类地形、20 秒回合、x/y ±1 m/s、yaw ±1.5 rad/s；楼梯单独限速。腿目标按 URDF 硬限位裁剪，轮目标按缩放后 ±44 rad/s 裁剪。新奖励移除了静止首次触地正奖励、未验证的对角镜像项和持续的 upright 正项；未加入强制抬腿奖励。

## 验证边界

机器人站姿和真实机械范围仍未验收。相对地面高度与额外髋惩罚仍是默认关闭的候选，观测噪声与域随机化也未开启；本轮不能证明步态更稳定、楼梯通行改善或实机可部署。
