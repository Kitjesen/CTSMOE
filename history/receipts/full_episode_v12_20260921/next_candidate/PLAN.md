# 下一候选：flat normal-reset explicit startup sampler

状态：**prepared not launched**。本目录只包含离线纯 tensor sampler、统计测试和本方案；没有接入 production、Isaac class、评估脚本或训练启动。

候选只改变 flat normal-reset cohort 的初次命令：选中 cohort 的 50% 使用十类显式原语（前、后、左、右、左/右纯 yaw、两组随机象限对角、前进转向、侧向转向），另 50% 完全沿用现有 `MixedCommand`。后续 command update 保持原 5 秒 sampler；episode 保持 25 秒。其他 terrain、奖励、V6 anchor 系数与 first1 窗口、4096 env、terrain allocation 均不变。

纯 tensor 合同在 `pure_start_sampler.py`：返回 `(goals[count,3], kinds[count])`；kind 0..9 与 `KIND_NAMES` 一一对应；kind 0..5 的方向符号固定，kind 6..9 的组合分量才做正负对称；非零分量范围为 `|vx|∈[.2,1.2]`、`|vy|∈[.2,.5]`、`|yaw|∈[.3,.8]`，未声明分量严格为零。`explicit_selection` 只对每个完整 learner cohort 初始化一次，随后用 `reset_selection(reset_ids, global_mask)` 求交，单环境 reset 仍继承全局选择。

测试只验证 shape、kind、分量边界、零结构、seed 重现和 50% selection；不验证 Isaac 环境行为，也不暗示 masked sample 变少会使 loss 权重变小。

倾向的受控源 checkpoint 是 V9 `44711`，计划候选 300 updates，与 V12 做顺序受控对比；不并行、不自动启动。执行前仍需等待当前原生梯度/时程结果；本候选尚未决定执行。

父代理复核：修复测试中的侧移转向分量预期与随机mask硬编码问题后，2026-09-21实际运行pytest退出码0，4 passed。该速度域是现有anchor掩码的选择范围，不代表V6在域内所有随机场景都已验收。
