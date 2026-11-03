# V21 runtime preparation

本目录只准备隔离运行时及一次零更新资格检查。不连接远端，不启动本地 GPU，不调用训练 driver，也不会在预飞通过后自动训练。原 V19/V20 文件均保持原样。

## 上传与构建

将本目录八个 payload 文件上传到 `/root/autodl-tmp/cts-omni-repair-v19-20260922/stair_coverage_v21_payload/runtime_prepare/`；将已审阅的 `coverage_patch/stair_transition_coverage.py` 上传到该 staging 的 `coverage_patch/`。八个文件是 `build_runtime.py`、`runtime_contract.py`、`preflight.py`、`run_preflight.py`、`verify_preflight.py`、`preflight_support.py`、`coverage_observation.py`、`SOURCE_IDENTITY.json`。

远端由主任务执行：

```sh
python /root/autodl-tmp/cts-omni-repair-v19-20260922/stair_coverage_v21_payload/runtime_prepare/build_runtime.py --coverage-module /root/autodl-tmp/cts-omni-repair-v19-20260922/stair_coverage_v21_payload/coverage_patch/stair_transition_coverage.py
```

`--root` 默认指向上述 V19 receipt 根，`--payload-dir` 默认脚本所在目录。目标固定为 `ROOT/stair_coverage_v21`，允许不存在或已存在但严格为空；不删除、不覆盖已有证据。构建失败留下部分目录供调查，不自动清理重试。

构建器读取实际远端 `ROOT/speed_brake_v20/PREPARED.json`，核验该运行时完整 86 项源码及额外 Python 文件不存在，然后逐项复制实际源码。原 86 项中只改 `train.py`，新增一个 coverage 模块，结果清单共 87 项。coverage 模块必须匹配已审阅版本的固定 SHA，创建目标前拒绝同名不同内容。`PREPARED.json` 保留原 86 项及新 87 项哈希；`RUNTIME_PREPARE_MANIFEST.json` 独立记录八个准备文件。运行前后同时验证 V20 源码、V21 源码、准备脚本和源 checkpoint。

`SOURCE_IDENTITY.json` 是本地 V20 `EVALUATION_33865_IDENTITY.json` 的逐字复制，所有脚本从该文件读取 SHA。源为 V20 `training_33174_to_34823/model_33864.pt`，已完成 33865 更新，common step 为 812760；没有手抄或重新推断 SHA。身份文件原始字节的 SHA 由已审本地文件程序生成并固定在 `runtime_contract.py`，因此替换 checkpoint 和身份 JSON 不能形成新的自洽授权来源。

## train.py 的限定插入

1. 在 `configure_mixed(...)` 的下一行导入并调用 `stair_transition_coverage.configure(cfg)`，早于 `Go2Env(cfg=cfg)`。
2. 原模型、PPO optimizer、student optimizer、地形等级/列、命令成员关系、课程、learner 分组及 allocation 全部完成原有恢复后，在 `run_receipt` 创建前调用一次 `snapshot(raw)`，写 `output/COVERAGE_GEOMETRY.json`。
3. 同一 snapshot 存入 `run.json["stair_transition_coverage"]`。剥离三段插入后必须逐字符还原原始 train 文本。

没有改奖励、命令、执行器、分配比例、等级课程、终止条件、恢复逻辑或学习循环。几何的授权范围仍是上楼 col8、下楼 col11；外圈台阶截成平台，原内区 riser/tread 不变，但整段总升高和外圈阶数减少，不能宣称整个楼梯难度完全相同。

## 预飞及独立资格

远端执行 `stair_coverage_v21/run_preflight.py`：取得现有全局 GPU 锁，并确认 GPU 无任务后，以 `--updates 0` 运行一次 4096 环境、3000 控制步预飞。源 profile 参数沿用 V20。输出和日志均排他创建，失败不自动重试。`READY_FOR_REVIEW.json` 仅标记等待独立核验。

随后单独执行 `stair_coverage_v21/verify_preflight.py`；只有它能够产生 `VERIFIED_READY.json`，状态仍为 `verified_not_training`。主任务另外负责是否进入训练、16+942 的预算以及训练后评估。

零更新证据包括：

- 使用原样复制的 `MutationGuard` 禁止 `alg.update`、`compute_returns`、双 optimizer `.step` 及 `runner.save`。
- 要求恢复的 runner iteration 为 33865，24 步 storage 初始为空；预飞前后模型及双 optimizer 全状态 SHA 相同，且独立核验与 checkpoint 三项真实状态的 SHA 相同。
- runner iteration 保持 33865，common step 从 812760 到 815760；没有 `.pt` 输出，源 checkpoint 字节和哈希不变。
- 保留 V20 实际速度上限覆盖、零命令/纯旋转/启停、两个 learner 采样、pose gate、lift、扭矩、边界及 allocation 资格。
- 复核实际有效奖励权重和 profile，拒绝追加 landing_motion 或更改平地比例；保留源来的 reset 驱动高度课程。因此预飞内既有高度权重随控制时钟继续变化，这不是新增奖励规则，也不是优化器更新。

程序不接受 `python -O`，以保证沿用的断言有效。当前本地检查不能证明 Isaac 接口、实际网格和机器人行为，必须由主任务执行远端预飞后读取真实证据。

## COVERAGE_GEOMETRY.json schema

记录发生在**恢复后且预飞/训练第一步之前**。关键字段：

```text
profile = stair_terminal_columns_v21
config.columns_by_kind = {stairs_up: [8], stairs_down: [11]}
config.generated_rows = [0..21]
config.active_original_rows = [0..9]
generated_cell_count = 880
selected_cell_count = 44
mesh_receipts = 880 条 row/column/kind/selected、before/after 哈希和 origin 证据
assignments.total = 4096
assignments.coverage_total / fraction
assignments.by_learner.teacher / student:
    total = 3072 / 1024
    coverage
    by_type.<九种地形>:
        total / coverage / original
        coverage_by_level = 22 项
```

被选 mesh 还含内区顶点、实际输出内区三角顶点和三角坐标的前后哈希、未改布尔证明、实际 cardinal 几何 profile、cap/center 高度及删除退化面数量。未选 mesh 的 before/after 哈希必须完全相同。两个 learner 均须存在上楼和下楼的原始及新增平台 cohort。源实际总覆盖由主任务的 `CHECKPOINT_TERRAINS_33865.json` 报告为 up8=87、down11=159，即 246/4096；本模块记录并校验实际恢复值，不把该期望硬写为观测。

预飞 FINALIZED 保留 `initial_coverage`，`settings.stair_transition_coverage` 为预飞末尾实际统计；mesh 证据必须相同。环境会按原课程在预飞中 reset 和改变等级，因此末尾等级覆盖可能不同，不能替代初始恢复记录。

`FINALIZED.coverage_observation` 使用初始 selected cohort，按 `teacher/stairs_up` 等四组记录实际 env steps、到过该状态的环境数及最长连续秒数：根节点进入 `[2.9,4.4]m` Chebyshev 外圈、四轮水平圆包络在该带内、已有 phase ready、外圈 phase ready、外圈 pose weight≥0.9。还记录外圈 phase/pose weight 最大值。只读奖励计算已有缓存，不调用 measure、不加 raycast、不推进任何课程或 recovery 状态。done/reset 帧被剔除，缓存缺帧打断连续时长并使独立资格失败。这些计数不证明接触支撑、完整机体 hull 覆盖或训练时间足够；原约4m距离终止可能令高速平台暴露极短。

## 本地验证

`python -m pytest -q runtime_prepare/test_runtime_prepare.py`：32 passed。

`ruff check runtime_prepare`：通过。测试不依赖 Isaac，不连接 SSH/GPU，包含 86→87 文件构建、严格空目录要求、三段插入可逆性、身份/源码损坏拒绝、完整及缺失网格证据、双优化器/clock/profile/reward/速度资格拒绝，以及五个禁止训练/保存入口的实际守卫测试。
