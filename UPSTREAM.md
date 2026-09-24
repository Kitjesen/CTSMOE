# 上游与许可
原项目：https://github.com/wertyuilife2/go2_rl_robotlab
原始迁移固定提交：`28b4516d22617b11aeaf8ead63cc00b0c0bcd1bd`，依据 history/initial_adapter/README.md。
服务器没有保留完整Git元数据，因此 vendor 是实际运行文件快照，不能冒称每个文件均与该提交完全一致；以 provenance 清单的 SHA256 为准。

CTS/MoE核心位于 vendor/upstream/source/rsl_rl/rsl_rl/ 的 modules/actor_critic_moe_cts.py、algorithms/moe_cts.py、runners/on_policy_runner_cts.py 及对应storage。原Go2任务位于 vendor/upstream/source/robot_lab/robot_lab/tasks/go2/。
保留源文件版权头。上游固定提交的许可证单独保存于 vendor/upstream/LICENSE；Thunder及本项目不因此自动获得同一种许可证。仓库保持私有；不为所有文件统一重授许可证。
