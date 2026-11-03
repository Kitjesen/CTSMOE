# NEWPOSE 高度奖励与方块课程候选

本目录是隔离代码，未覆盖 19538 正式配置、检查点或奖励。它不依赖旧的 artifact 路径；将本目录作为包放在新训练入口的 Python 路径中即可导入。

## 接口

在创建环境之前，调用 `height_curriculum.configure(cfg, gradual_boxes=False, height_weight=0.0, height_options=None)`。

- 全部默认参数保持原配置；两个候选可独立开关。
- `gradual_boxes=True` 只替换 `cfg.scene.terrain.terrain_generator.sub_terrains['boxes'].function`，并关闭地形缓存。宽度、布局、比例、平台和原生网格构造保留。曲线复用 20261102 已做固定策略测试的 gradual 版本：difficulty=0/0.3/1 时幅值分别为 0.01/0.046/0.20 m。随机高度仍为正负幅值，相邻落差最高接近两倍幅值。曲线修改会改变中高等级的实际几何；不能将相同 L 值当作相同难度。
- `height_weight<0` 时必须显式提供 `HeightOptions(target_height, tolerance, max_reference_speed, max_reference_lag, min_valid_fraction=1.0)`。没有隐式“最优”高度或权重。正权重、非有限值、无效参数被拒绝。
- 高度项用 `height_scanner` 的局部 x ±0.20 m、y ±0.15 m 射线裁剪，要求 yaw 对齐，传感器至少每个策略步更新一次。读取 `robot` 的 base z；接入旧 Tensor 或 Isaac Lab 3 ProxyArray `.torch`，不涉及四元数顺序。
- 每步地面均值限速变化；缺测或参考相对当前扫描滞后超过指定阈值时该项为 0。代价为 `max(abs(base_z-reference_z-target_height)-tolerance, 0)^2`，由 Isaac RewardManager 施加负权重和 dt。
- 使用 `ManagerTermBase.reset(env_ids)` 清理对应环境的参考状态，避免 terrain/reset 的高度跳变污染新回合。`reference.active`、`lag`、`relative_height` 和 `ground_z` 可用于运行日志。

## 已做验证

- 本地 `python -m compileall -q .../height_curriculum` 通过。
- `audit_reference.py` 的独立 NumPy 自检通过，并在 8 条历史官方 IsaacLab 轨迹的离线重建扫描均值上回放：示例限速 0.5 m/s、lag 门槛 0.04 m，使 20 ms 最大参考变化不超过 1 cm，屏蔽比例为 0～8.62%。这不是在线传感器或闭环训练。源 NPZ 哈希和逐例结果在 `REFERENCE_REPLAY.json`。
- `python -m unittest height_curriculum.test_height_curriculum -v` 提供 9 项 Torch CPU 测试：高度容差、世界高度平移不变性、缺测、部分 reset、阶沿跳变及恢复、配置输入、方块曲线/随机流/原配置不变、禁用无改动、Tensor/ProxyArray manager 接口。当前本地 Python 无 Torch，这组测试由主任务在已有 CPU Torch 环境执行，结果需另行确认。adapter 测试使用 manager stub，不能替代 Isaac Lab 3 原生实例化。

## 验收边界

限速参考并不是支撑面估计，可能在持续坡面或跨阶时滞后，且屏蔽期减少高度学习信号。缺测也可能被策略利用。必须记录有效率/lag，验证相对支撑轮的实际机身高度、腿净空和碰撞，并对平地、上下楼 L6/L9 做同条件基线对照；不能只用奖励数值下降宣称改善。方块候选已有低等级固定策略改善，但 L9 仍未解决。

未运行 Newton/Isaac Lab 3 物理仿真、训练、实机或重新录制视频；不能宣称已修复机身下沉或腿碰。
