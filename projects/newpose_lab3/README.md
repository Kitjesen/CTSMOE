# Thunder NEWPOSE：Lab 3 从零重训

2026-11-03 决定：继续使用 Thunder 四轮腿机器人，复用 Go2 `lab3.0` 的 CTS+MoE 框架，在 Isaac Lab 3 / Newton 中建立独立实验，目标为 **0 → 20000 次参数更新**。旧策略仅作对照，不继承其权重、优化器或地形课程进度。

**当前是代码和资产基线，尚未启动从零训练。** `scripts/launch.py` 和 `compat/runner.py` 仍保留之前的旧检查点回放/续训适配行为；它们不是已经完成的新训练入口。机器人、默认站姿、动作、奖励与课程将按顺序核对后再改。

## 代码与资产

| 路径 | 内容 |
| --- | --- |
| `thunder_lab3/` | 八类地形、L0–L9、命令、奖励与任务注册 |
| `task_port/`、`runtime/` | Thunder 模型导入、16 个动作、观测与默认站姿 |
| `teacher/` | 教师接触/动力学输入及镜像审计辅助代码 |
| `height_curriculum/` | 相对地面高度与渐进方块候选 |
| `compat/` | 旧模型回放、导出及兼容性检查，保留作对照 |
| `scripts/` | 原有回放入口及不更新策略的原生环境探针 |
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

## 验证边界

这次入库只核对文件清单、源码语法、URDF 网格引用与来源哈希；不会启动训练。之前的迁移实验建立了接口与回放证据，但不能证明新策略更稳定或楼梯碰撞已解决。新的从零入口、机器人参数和新奖励仍待实现/验收，不能直接运行旧续训命令作为新训练。
