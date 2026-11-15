# Thunder NEWPOSE · Isaac Lab 3

使用自有 Thunder 四轮腿机器人，复用固定版本 Go2 `lab3.0` 的 CTS+MoE，在 Isaac Lab 3 / Newton 中从零训练，累计目标为 **100000 次参数更新**。按 2026-11-03 最新要求采用 Go2 速度课程，保留 Thunder 地形难度上限（楼梯 25 cm、坡面 40°）。

**2026-11-03 已启动 16384 环境、从零到 100000 次更新的正式训练。** 固定目标站立和训练入口检查已通过，学习后的步态尚未验收；承重高度偏差约 49 mm 仍待改善。旧 19538 策略只作回放对照；新任务不继承它的权重、优化器或地形进度。启动快照与迁移短测分别记录，见[验证记录](docs/validation.md)。

## 目录

```text
source/thunder_lab3/
├── assets/             # URDF、21 个 STL、关节约定和机器人配置
├── tasks/thunder/      # 环境、地形、MDP、CTS+MoE 配置与任务注册
├── training/           # 官方 Runner 的计数、超时、续训和导出适配
└── legacy/             # 旧 19538 检查点转换及历史状态读取
scripts/
├── rsl_rl/launch.py    # 统一训练、续训、回放入口
└── probe_native.py     # 无优化器的原生环境检查
tests/                 # 测试与旧配置 fixture
tools/                 # 独立审计和旧模型验证工具
upstream/go2_rl_lab/    # 固定提交的官方 Git 子模块
docs/                  # 配置差异、验收证据和来源清单
archive/v33/           # 原 V33 工程，完整保留，仅作历史参考
```

当前任务集中在一个 `thunder_lab3` 包内。官方网络、PPO、蒸馏和训练循环仍由上游提供；`training/` 仅保留 Thunder 所需的边界适配。旧实验不作为当前运行入口。

任务配置`env_cfg.py`、`rsl_rl_cfg.py`同级，不再设`agents/`目录。高度参数类型、地面参考及奖励接入统一放在已有奖励模块`mdp/terms.py`，目标值和权重直接在`env_cfg.py`中设置；方块难度曲线与其他地形统一放在`terrain.py`，不再单设高度或方块代码文件。

当前目录精简版尚未替换服务器运行源码；已启动的L0～L5训练继续使用冻结提交`f4f1030`。模块路径及源码指纹发生变化，现有检查点仍须使用原冻结版本续训，不能直接在此精简版绕过恢复校验。验证范围见[验证记录](docs/validation.md)。

## 环境与安装

```bash
git clone --recurse-submodules --branch newpose-lab3-from-scratch https://github.com/Kitjesen/CTSMOE.git
cd CTSMOE
# 在已经配置好的 Linux Isaac Lab 3 / Newton 环境中安装本包。
uv pip install --python /ABS/IsaacLab3/.venv/bin/python --no-deps -e .
```

固定依赖：Isaac Lab **3.0.0**、Newton/MJWarp、Go2 自带 **RSL-RL 3.3.0**，版本及提交见 [PROVENANCE.json](PROVENANCE.json)。适配验证使用 Python 3.12；不能用通用 RSL-RL 替换 Go2 的 CTS+MoE 实现。入口使用 standalone Newton，不代表完整 Kit/PhysX 环境也通过了验证。

仓库包含源码与 Thunder 网格，不包含 GPU 运行环境、检查点、视频、轨迹、生成的 USD 或服务器凭据。若需要上游 Go2 的 LFS 资产，在 `upstream/go2_rl_lab` 执行 `git lfs pull`；Thunder URDF/STL 直接随仓库保存。上游许可证保留在该子模块及其 `source/rsl_rl/LICENSE`，Thunder 资产不因此获得上游许可证授权。

## 运行

以下命令均从仓库根目录执行。`launch.py` 默认只打印计划，添加 `--execute` 才启动模拟器或更新策略。

```bash
THUNDER_URDF=source/thunder_lab3/assets/thunder/urdf/thunder_v4.urdf

# 从 0 开始的两次更新检查；不加载检查点。
python scripts/rsl_rl/launch.py train --urdf "$THUNDER_URDF" \
  --num-envs 32 --updates 2 --seed 42 --run-name smoke

# 同版本、同任务的明确续训；updates 表示额外更新数。
python scripts/rsl_rl/launch.py train --urdf "$THUNDER_URDF" \
  --num-envs 32 --updates 1 --seed 42 --resume --checkpoint /ABS/model_2.pt

# 官方 Newton GL 录像，600 个控制步为 12 秒。
python scripts/rsl_rl/launch.py play --urdf "$THUNDER_URDF" \
  --checkpoint /ABS/model_3.pt --num-envs 4 --terrain stairs_up --level 6 \
  --video --video-length 600 --execute

# 新任务原生探针，不创建优化器；输出目录必须尚不存在。
python scripts/probe_native.py --fresh --urdf "$THUNDER_URDF" \
  --output results/native_fresh --num-envs 32 --steps 1002 --visualizer none --device cuda:0
```

训练任务为 `Thunder-Newpose-Newton3-Fresh-v0`。回放可选 `flat`、`stairs_up`、`stairs_down` 和固定 L0–L9；这些选项不改训练课程。旧 19538 回放增加 `--legacy`，不允许用它续训新任务。播放先复制并校验检查点到独立 `results/play_*`，导出与录像写在副本旁；训练日志在 `logs/`，转换缓存在 `generated/`。

续训会核对实际配置、源码、URDF、环境数、地形、两个 Adam 及已完成更新数，并恢复课程和随机数状态后重新开始回合；不恢复中途物理轨迹。`model_N.pt` 的 N 是已完成更新数，新任务累计不得超过 100000。**当前配置变更了姿态、速度课程、步长、奖励、噪声与随机化；此前诊断检查点需使用对应源码回放，不能直接接续到新配置。**

候选参数通过重复 `--env-override FIELD=JSON` 指定，只接受 `gradual_boxes`、`height_weight`、`height_options`、`hip_extra_weight`。默认值和验收边界见[环境迁移说明](docs/migration.md)、[高度与方块说明](docs/height.md)。

## 机器人与训练边界

- 12 个腿位置动作、4 个轮速动作。当前 URDF 四髋限位 ±0.41 rad；前小腿 FR `[0,3.14]`、FL `[-3.14,0]`，后小腿 RR/RL `[-3.14,3.14]`，都不是连续关节。文件限位尚未等同于实机机械验证。
- 新任务使用 `assets/contract.py` 中的 `A_POSE` 两位小数版：按用户要求向零截断，左右关节角绝对值严格相等。前腿 thigh/calf 为 ±0.68/±1.65，后腿为 ±2.53/±1.53。它是 Go2 风格 A 的近似姿态，未继续宣称无限精度角度匹配。URDF、PD、质量惯量未改；B 仅保留作历史对照。
- 默认 **16384 环境**；短测请显式 `--num-envs 32`。10 级、8 类地形，初始等级与 Go2 一致，随机采样 L0～L5，后续按课程在 L0～L9 升降；25 秒回合，5 ms × 4、50 Hz 控制。Go2 全速度课程从 x/y ±0.5、yaw ±1 开始，45000 轮后平地 x -4.5～4.2、y ±1、yaw ±3；上下楼梯 x/y ±1、yaw ±1.5。每轴范围还受官方耦合规则约束。腿目标按 URDF 限位裁剪，轮目标限幅 ±44 rad/s。
- 接入官方动态命令/站立课程，裁剪后的实际命令参与预算；两位小数 A 相对地面几何高度目标同步为 0.391415 m、容差 2 cm，高度权重由 -1 到 -10（5000 轮），垂直速度权重由 -2 到 0（1500 轮）。探索参数 scalar、奖励裁剪 ±10、每 500 轮保存；24 步 rollout、网络和 PPO 参数沿用官方。
- 已接入 Go2 共同奖励、二阶动作平滑和独立髋惩罚、学生观测噪声、质量/摩擦/增益/零偏/推扰/COM 随机化，以及 Thunder 专用 16 动作左右镜像增强。地形升降级使用 Go2 距离公式；轮部奖励和接触终止保留。没有强制抬腿奖励。随机化与镜像的适配范围及验收边界见[官方配置对齐](docs/alignment.md)。

测试命令和原生证据统一见[验证记录](docs/validation.md)；观察历史、超时处理及旧模型转换细节见[训练与旧模型边界](docs/legacy.md)。承重稳定性、腿部碰撞、L6/L9 通行和实机部署均需单独验收，配置对齐不构成运动效果改善。

## Lope

Lope 是并列的 12 自由度固定足任务，不替换上面的 Thunder 正式训练。任务 id 是 `Lope-Newton3-Fresh-v0`。执行器、12 自由度动作比例和 RS02 保持 Lope 硬件数值。髋关节内收 0.08 rad，出生高度 0.2740510271690917 m，高度目标 0.2627175562793766 m。命令使用官方 mixed Go2 的未缩放速度课程，平地后期 x 为 −4.5～4.2 m/s。同一任务里约 20% 的重置以倒地姿态开始，并用 6 秒未站起超时代替瞬间的机身触地终止。正式训练为 16384 环境、100000 次更新、25 秒回合。楼梯为 5–25 cm，坡面为 0–35°，坡度按高宽比写入 `tan(35°)`。细节见 [docs/lope.md](docs/lope.md)。

```bash
python scripts/rsl_rl/launch_lope.py train \
  --urdf source/lope_lab3/assets/lope/urdf/lope.urdf \
  --num-envs 32 --updates 2 --seed 42 --run-name lope-smoke
```

不加 `--execute` 时只打印计划。
