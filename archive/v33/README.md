# CTSMOE — Thunder CTS + Mixture of Experts

> **2026-11-03：Thunder NEWPOSE / Isaac Lab 3 从零重训任务**
>
> 本分支 `newpose-lab3-from-scratch` 的代码和核对说明见 [projects/newpose_lab3](projects/newpose_lab3/README.md)。
> 新实验目标为 0 → 20000 次参数更新；迁移代码与独立短测试位于该项目目录，长训练尚未启动。
> 以下内容是保留的 V33 历史归档，不代表新任务已经通过验证。

私有研发归档：保存当前 V33 实现和可找到的父版本/实验分支证据。归档日期 2026-09-25。

## 从哪里开始
- [当前配置](docs/current_v33.md)、[环境与运行](docs/setup.md)
- [模型继承与训练历史](docs/training_history.md)
- [奖励调整](docs/reward_history.md)、[地形调整](docs/terrain_history.md)、[命令调整](docs/command_history.md)
- [已知问题](docs/known_issues.md)、[来源与复现边界](docs/reproducibility.md)

`src/thunder_v33/` 是服务器 V33 冻结代码的逐文件副本，未为目录美观重构导入。
`vendor/upstream/` 是实际使用的上游代码；`vendor/thunder/` 是实际机器人适配与资产。
`history/receipts/` 保存本地实验源码、计划、配置和评估摘要；`history/server_sources/` 保存服务器仍存在的旧源码快照。失败尝试也保留，不能直接执行旧监督器。
`provenance/` 保存文件清单、哈希和模型引用；权重不在 Git 中，未删除或替换原检查点。

当前训练为 8192 环境，原有8类+新增6类，高台环境为0。等级和通过率不同；本库不是实机发布包。

## 使用边界
历史绝对路径仅用于溯源，不是其他机器上的有效路径。禁止照抄历史 PID/STOP/自动续训入口。
当前入口提供显式检查点续训命令；首次使用先在自己的 Isaac 环境验证。CPU测试不等于GPU物理验收。
本次发布保留实际代码布局，后续模块拆分应独立验证。上游署名与许可见 [UPSTREAM.md](UPSTREAM.md)。
