# 训练接口与旧模型边界

`thunder_lab3.training.runner.ThunderMoeCtsRunner` 继承固定Go2的`rsl_rl.moe_cts.runner.MoeCtsRunner`，复用网络、PPO、蒸馏、存储、日志和训练循环。Fresh开启Thunder专用16动作镜像；旧回放仍关闭，Go2的12动作映射被拒绝。RoboGauge保持关闭。

## 当前训练依赖

环境提供 `policy`（530）、`single_obs`（53）、`critic`（309）TensorDict。历史是十个完整单帧按时间排列，当前帧在最后，reset 后用第一帧重复填充；不能对六个观测项分别建立 term-major 历史再混用。生产历史由环境管理器维护。

环境必须声明 `newpose_history_layout='frame_major_current_last'`、`compute_final_obs=True`、`newpose_timeout_bootstrap='terminal_observation'`，使用无限时域的时间限制价值补偿。

超时适配用 transition 的前一步历史加终止单帧重建终点历史，按教师/学生样本选择正确的 critic，加入 `gamma*V(s_T)`。同时真正终止的样本不补偿；处理后移除 `time_outs`，避免上游再次补偿。缺少终点观测时报错，不使用 reset 后观测替代。

新任务从随机初始化的0次更新开始，格式为`thunder_newton_fresh_v1`。保存初始`model_0.pt`、两个优化器、课程/RNG、任务/资产/源码指纹。只对成功的`alg.update()`计数，保存`iter=completed_updates`，修正上游zero-based文件名和恢复入口的`iter+1`。续训恢复自适应学习率标量；新任务预算100000，历史legacy预算仍20000，不从文件名猜更新数。

恢复要求同配置、同环境数和相同地形布局；课程和 RNG 在新回合 reset 前恢复，中途物理轨迹不恢复。目录调整改变函数模块名和源码指纹，整理前的诊断检查点可用于回放，不能绕过签名继续同一次训练。

## 旧 19538 回放

旧正式检查点 SHA-256 为 `8db552f69ab35105f737e74be9477c82e2d539a5d3984b2c32e41c1e992c226e`。转换逻辑在 `thunder_lab3.legacy.teacher_checkpoint`，历史环境状态读取在 `thunder_lab3.legacy.environment_state`；当前入口仅允许 `play --legacy`。

旧模型到 Newton 会移除旧 dynamic-friction 输入 272、增加 36 个教师特权输入，教师从 274 维变为 309 维。学生与共享 actor 权重保持，两个 Adam 历史保留并调整教师/critic 对应矩；教师预测可能改变，不是教师前向等价或连续物理状态的迁移。

旧 terrain levels/types、计数和 Python/NumPy/Torch/CUDA RNG 保留在带版本的 `environment_state` 中，但不假定可直接恢复到 Newton。旧后端课程映射和原生验收未通过，旧谱系的正式续训仍拒绝执行；没有可绕过的配置开关。CPU 测试中的合成环境专用子类不适用于机器人训练。

## 导出和验证工具

JIT 接收完整 `batch × 530` 历史，输出 `batch × 16` 动作；ONNX 使用上游固定形状导出，输入 `1 × 530`、输出 `1 × 16`。两者复用上游 encoder 和 actor，不在内部维护历史。导出元数据使用明确的已完成更新数；迁移改变环境数时，不编造累计环境步数。

在配置好固定上游依赖、安装本包后，可从仓库根目录执行：

```bash
python -m tools.legacy.verify_bridge --source /ABS/model_19538.pt \
  --old-policy /ABS/policy_19538.pt --report results/legacy_cpu.json --exercise-update --onnx
python -m tools.legacy.verify_timeout_runner --source /ABS/model_19538.pt --report results/timeout.json
```

这些工具验证旧学生推理、未变权重、Adam、历史、计数、预算和超时补偿；优化器更新使用合成观测，不推进正式训练。可选 ONNX 对照还需要 onnxruntime 和上游导出所需的 onnx/onnxscript，只装在隔离的新环境。

当前从零训练的回归统一在 `tests/`。测试结果和原生验证范围见[验证记录](validation.md)，工具定义不等于测试已经通过，也不能代替站立或楼梯验收。
