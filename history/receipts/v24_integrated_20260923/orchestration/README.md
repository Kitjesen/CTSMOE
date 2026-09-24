# V24 有限训练入口：先单因素修髋内夹

本目录只准备代码和门控；截至编写时未启动 V24 GPU 训练。V23 源码与已运行阶段保持冻结。要把整个 `v24_integrated_20260923` 复制到独立远端目录，保留 `code/` 与 `orchestration/` 兄弟层级，不能覆盖 V23。入口仅支持 V23 配对评估过的两个候选源：

- 34823 轮 V21 检查点：训练时沿用原髋默认角；已在 V23 的 ±0.12 rad 仿真配置下作为冻结对照被重新评估。重新训练会应用 V23 的 ±0.12 配置，因此不能把其旧训练史表述为“始终用 ±0.12 训练”。
- 35721 轮 V23 检查点：已在 ±0.12 rad 配置下继续了 898 更新。只有 V23 的 604 个固定配对案例确认未显著损害运动能力时才选择它。

选源必须由操作者审阅 V23 的九地形、25 秒平地、楼梯末端和 15 厘米阶梯的学生/教师轨迹后，在 `orchestration/SOURCE_REVIEW.json` 写入以下明确决定；`prepare.py` 不自动选择最近检查点：

```json
{
  "decision": "eligible_for_isolated_hip_request_test",
  "source_path": "/absolute/reviewed/model_XXXXX.pt",
  "source_sha256": "64 lowercase hex characters",
  "source_completed_updates": 34823,
  "v23_evaluation_path": "/root/autodl-tmp/cts-omni-repair-v19-20260922/hip_default_v23_20260923/evaluation_35721/FINALIZED.json",
  "v23_evaluation_sha256": "64 lowercase hex characters",
  "behavior_findings": "State the actual posture, stairs, platform and actuator findings that justify the source.",
  "real_robot_release": false
}
```

把 `source_completed_updates` 设为实际选定的 34823 或 35721；脚本会从 V23 配对评估的哈希账本核验该源。V23 已经消耗的 898 更新不会因为换回早期源而退还。原 31831 起的总 10000 更新预算已花 4000（含废弃的 V22 102 更新和较早丢失的 8 更新），V24 只用 1000，剩余 5000。

在 V23 评估结束、GPU 空闲且脚本与代码定稿后执行：

```sh
python orchestration/prepare.py
python orchestration/driver.py
```

`prepare.py` 只读取源与评估证据，然后冻结源码、评估协议、源检查点哈希。`driver.py` 有一次性标记、唯一流水线锁、GPU 占用与空间核查：先跑 3000 控制步、零优化器更新的髋惩罚预飞，独立核对模型与教师/学生两个优化器哈希、真实源轮次、原地形分配；之后更新 16 轮，保存并再次恢复双优化器和地形状态；再更新 984 轮，到选定源轮次加 1000 后停止。用冻结的 V23 **同一行为评估协议**对源/候选学生和教师运行 332 个九地形、80 个 25 秒平地、96 个楼梯末端、96 个 15 厘米台阶固定案例。评估的 V23 代码并不运行新奖励，只观察训练出来的动作，这让源与候选在同一环境下可比。

`BEHAVIOR_COMPARISON.json` 分开给四髋原始内夹请求超过 0.369 rad 的量、目标逼近软限的时间、实际角接近 0.40 rad 的 p95 与持续时间，另给 15 厘米阶梯和九地形通过/物理失败、腿轮扭矩与转速、触地峰值、跟踪、楼梯末端世界竖直速度及高台固定案例实际成功高度。**原始请求变小不自动算通过**。所有这些是固定仿真案例；不能推出随机成功率或实机可用。当前资产关闭自碰撞，所以“仿真没有腿碰撞”也不能证明实机不会卡腿；原始实机故障码和控制日志仍缺。

技术错误和行为退化都停下，保存日志、最后观测到的更新与检查点，不自动重试、延长预算或切入混合楼梯。候选完成后的 `pipeline_status.json` 是 `complete_pending_behavior_review`，不是实机放行。

混合楼梯只允许**单独的零更新诊断**，不能由此入口训练；在 GPU 空闲、源已冻结时手动运行：

```sh
python orchestration/qualification.py --mode mixed_diagnostic
python orchestration/verify_qualification.py --mode mixed_diagnostic
```

它核对首次 reset 后 V21 选中楼梯的直行、前进＋Yaw、前进＋侧移组数，以及真实地形力传感器张量、经验证的轮碰撞包络和机身碰撞壳体。轮下接触位置是保守几何推断，不是实测接触点。V23 原评估器不能把新混合楼梯组当直行测试；后续若单独训练混合命令，必须再做针对该组的独立固定轨迹协议。
