# Thunder CTS+MoE：平地默认关节姿态、从零初始化准备

状态：**本地代码已修改并完成 CPU 验证；未连接服务器、未提交或启动新 GPU 训练**。

来源任务：codex://threads/01a0a831-8a25-7951-9c07-62090953379a。
授权范围：开始调整从零训练配置，并使用平地实验的默认姿态。新正式更新预算尚待用户登记。
没有修改 V23/V24/V25 的冻结代码、训练进程或课程队列。

## 默认参考角（rad）

| 腿 | Hip | Thigh | Calf |
|---|---:|---:|---:|
| FR 右前 | -0.12 | -0.93 | +1.96 |
| FL 左前 | +0.12 | +0.93 | -1.96 |
| RR 右后 | +0.12 | +0.93 | -1.96 |
| RL 左后 | -0.12 | -0.93 | +1.96 |

轮初始参考角为 0，默认关节速度为 0。按关节名称匹配，不假设 USD 内部顺序。
这些是默认参考，不是目标角锁定或关节硬限位。重置随机化仍可让实际姿态偏离参考。
关节相对观测、动作默认偏置、引用 default_joint_pos 的姿态代价随同一参考改变。

只迁移关节姿态，不迁移平地 PPO、53/56 维接口、增益或奖励包。
保持 CTS teacher/student：student 单帧 53、10 帧历史 530、critic 291、动作 16。
保留粗糙任务的基座名义高度 0.46m、地形相关重置及高度奖励；未把平地
0.451915435999m 固定高度直接套在坡面/台阶上。
腿标称 KP120/KD8、腿动作 scale0.25、轮 scale5 均沿用粗糙任务。
新入口显式选用已存在 rs0402_transient_v4：继承 12rad/s 目标变化率和
原物理/软限位，而不是把 Hip 默认 0.12 当作限幅。该配置不代表硬件安全认证。

## 实现

- `../../default_pose.py`：独立 profile、配置和运行时默认位置/速度核验。
- `../../train.py`：`--default-pose flat_reference_v1`；保存 pose contract；恢复时校验，
  不允许旧策略在未迁移情况下悄悄换默认姿态。新建策略会验证更新计数为0、
  common step为0、PPO和student优化器状态为空。
- `../../train_scratch.py`：禁止 resume/预训练入口，正式更新数必须显式给定，输出必须新目录。
- `../../policy_contract.py`：旧评估入口默认拒绝不同姿态导出；新评估必须先应用
  matching pose，再以 `expected_pose="flat_reference_v1"` 加载，不能只换模型文件。

## 课程边界（必须决定，尚未完成）

现有最新九地形入口依赖成熟 checkpoint 的分组、地形层级、坑高、命令课程等状态；
不能删除 resume 参数就声称支持九地形从零训练。本轮没有伪造这些成熟状态。
目前独立入口只在用户显式选择 `--terrain-course original8` 时允许原始八地形从零。
如果用户选择保留九地形目标，仍需实现和验证独立的冷启动课程，不应启动此八地形入口冒充它。
原始八地形模式从零指网络/优化器/训练计数从零，并不保证所有初始 terrain level 都为0。

CPU dry-run 示例（8 是短测候选，不是用户正式预算，不启动 SDK）：

```text
python train_scratch.py --root /root/autodl-tmp/NEW_CTS_ROOT --output /root/autodl-tmp/NEW_RESULTS/smoke --updates 8 --num-envs 64 --terrain-course original8 --dry-run
```

`NEW_CTS_ROOT` 是待核实安装目录，不是已经部署的路径。运行环境仍需要固定版本
CTS upstream/Thunder source，不能假设原平地的 robot_lab 安装就是 CTS 环境。

## 验证与尚未验证

CPU pytest：`test_default_pose.py` + `test_policy_contract.py` 共 **68 passed**。
覆盖精确平地角度、资产复制隔离、保留增益/动作尺度/基座高度、关节乱序、
checkpoint继承/拒绝悄悄换姿态、真实 TorchScript 导出姿态匹配、禁止恢复、显式预算。
Ruff 检查新增模块及修改的 train.py/policy_contract.py 通过。
现存 TorchScript API 有弃用警告，没有在本次顺带更换导出格式。
未做 Isaac 集成、GPU 更新、摆姿态/触地/关节偏置随机化后的实测审核；CPU通过不等于可正式开训。
正式启动前还需：用户确定地形课程和预算、独立数据盘源/结果目录、GPU锁与容量核验、
短测真实退出/初始化与导出一致性检查。旧任务预算不自动沿用。
