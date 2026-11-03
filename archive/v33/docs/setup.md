# 安装与续训
需要与现有服务器匹配的NVIDIA GPU、Isaac Sim / Isaac Lab、PyTorch与rsl_rl依赖。准确环境快照见 provenance（如缺项必须先补齐，不凭猜测锁版本）。
仓库不捆绑Isaac Sim/Isaac Lab二进制；请在已配置Isaac环境的Python中运行。
`vendor/`是训练入口的runtime root：包含upstream/source/rsl_rl、upstream/source/robot_lab、thunder/source/doso_train。

```bash
python scripts/train_v33.py --checkpoint /your/model.pt --output /new/run --updates 16 --dry-run
```
确认输出命令和检查点来源后移除`--dry-run`。该入口明确要求V33兼容8192环境检查点；旧父版本需要对应迁移，不可直接套用。
历史orchestration是运行证据，不是跨机器通用启动器。模型/双优化器、课程、环境索引和预算必须共同恢复。
训练期间不并行启动另一GPU录制/评估任务。不要使用未知来源pickle检查点。
