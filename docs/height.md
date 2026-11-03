# 高度参考与渐进方块

相关实现统一在 `source/thunder_lab3/tasks/thunder/`：`boxes.py` 负责渐进方块；`mdp/height.py`、`isaac_height.py`、`height_config.py` 负责高度参考与奖励接入。环境在创建前通过 `apply_candidate_features()` 统一应用候选，避免绕过配置校验。

新任务默认启用渐进方块与高度候选：`height_weight=-1`，5000 轮线性到 -10；A 向零截断至两位小数后，最低轮触地几何高度目标同步为 0.39141468661229084 m，容差 0.02 m，参考速度 0.5 m/s、最大滞后 0.1 m、有效射线比例 0.8。该值不是承重实测高度；A 在水平机身下也不是四轮共面。`height_weight=0` 同时关闭该奖励和对应权重课程。旧 19538 回放配置仍关闭该项。

## 方块

`gradual_boxes=True` 替换方块生成函数并关闭地形缓存，保留格宽、布局、比例、平台及原生网格构造。difficulty=0/0.3/1 时幅值 g 分别为 0.01/0.046/0.20 m；格高仍在 `[-g,+g]` 取样，相邻落差最高接近 `2g`。45 cm 是格子边长，不是高度。

这条曲线改变了相同等级的实际几何。历史固定策略测试显示降低起步幅度改善了低等级通过，但未解决 L9；不能将几何简化当作策略能力提高。

## 高度

启用负的 `height_weight` 必须明确给出 `HeightOptions(target_height, tolerance, max_reference_speed, max_reference_lag, min_valid_fraction=1.0)`，不存在隐式最优高度或权重。正权重、非有限值、无效参数被拒绝。

- 对 yaw 对齐的 `height_scanner` 裁剪局部 x ±0.20 m、y ±0.15 m 射线，传感器至少每个控制步更新一次；读取机器人 base z，兼容 Tensor 与 Lab 3 ProxyArray。
- 每步扫描地面均值按速度限制变化。缺测或参考滞后超过阈值时，代价为 0；有效时为 `max(abs(base_z-reference_z-target_height)-tolerance,0)^2`，RewardManager 再乘负权重与 dt。
- `reset(env_ids)` 清理对应环境的参考状态。应记录 active、lag、relative_height 和 ground_z，检查 reset、坡面及跨台阶时的参考行为。

## 验证与限制

测试在 `tests/test_height_curriculum.py`，覆盖容差、世界高度平移不变性、缺测、部分 reset、台阶跳变与恢复、输入校验、方块曲线/随机流/原配置隔离及 Tensor/ProxyArray 接口。运行和结果见[验证记录](validation.md)。

独立历史工具 `tools/audits/height_reference.py` 曾在 8 条官方环境轨迹的离线重建扫描均值上回放：示例限速 0.5 m/s、lag 门槛 0.04 m，将 20 ms 参考变化约束到 1 cm 内，屏蔽比例 0～8.62%。这不是在线传感器或闭环训练证据，原始报告保留在实验记录中。

重放需要自行提供 8 个 `*_series_v2.npz`，输入不随仓库发布：`python tools/audits/height_reference.py --source /ABS/traces --output /ABS/report.json`。

限速扫描均值不是支撑面估计，持续爬坡或跨阶时可能滞后；屏蔽会减少学习信号，缺测还可能被策略利用。必须同时检查支撑轮、实际离地高度、腿净空、碰撞及平地/L6/L9 通行，不能仅凭代价下降宣称机身下沉或腿碰已解决。
