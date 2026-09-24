# Thunder 腿间净距：只读几何诊断

本工具把 V23/V24 固定案例评估记录的每步实际关节角带入与评估 manifest 哈希一致的 Thunder v4 URDF，计算**URDF 碰撞体的预测有符号距离**。它不加载策略、不运行 Isaac、不更新优化器、不修改训练代码或检查点。正值表示两个 URDF 碰撞体分离；负值表示碰撞体几何重叠；零值可能是接触边界或距离算法的退化情形。所有距离单位为米。

V23 最终评估本地有 `stairend_15cm_manifest.json` 和髋指标汇总，但**没有逐帧 `traces.npz`**。评估脚本 `reference/assessment/evaluate_hip_platform.py` 在远端评估输出目录保存 `traces.npz`，其中 `joint_pos` 为 `[step, case, joint]`，manifest 中有 `joint_names`、`cases[*].end_step`、`history`、`urdf_sha256`。因此当前本地材料不能得出 V23/V24 的真实逐帧腿间净距，也不能判定这次实机故障原因。取得对应评估轨迹后可运行：

```powershell
& 'C:\Users\99563\miniconda3\python.exe' .\probe.py `
  --manifest ..\reference\v23_final_assessment\stairend_15cm_manifest.json `
  --traces D:\path\to\matching\traces.npz `
  --output .\stairend_15cm_clearance.json
```

在本目录运行；把 `--traces` 换成与 manifest **同一次评估**的文件。依赖 Python、NumPy、MuJoCo、PyBullet。默认逐帧分析；`--stride N` 可加快大轨迹，但可能漏掉步间最小值，结果保留 stride 和实际抽样帧数。脚本拒绝非零训练/优化器更新、URDF 哈希不一致、关节/案例维度不一致和无效 `end_step`。输出按案例分列 `inter_leg`（不同腿之间）、`leg_body`（全腿与机身）、`leg_body_distal`（大腿、小腿、轮与机身）；含最小距离、发生步数、碰撞体所属 link、1% 分位与抽样帧占比。源与候选仅在地形、等级、速度、方向、停车模式、教师/学生等固定案例键唯一对应时生成差值；正差值表示候选的最小几何净距更大。

几何模型来自 `wave_reset_20260919/thunder_v4.urdf`，SHA-256 为 `83ba02ca52863238ce4050fff7c122fb159e381fb106ba31e39e44bfc0b66af3`，与本地 V23 楼梯评估 manifest 一致。它导入 16 个转动关节、38 个盒体/圆柱碰撞体及 URDF 的 link/joint 原点和轴；全局机身姿态对同一机器人的内部距离没有影响，故根部固定。MuJoCo 只计算关节正向运动学，Bullet 对 URDF 碰撞体进行最近距离查询，没有执行物理步进。曾发现 MuJoCo 某些明显分离的圆柱—盒体直接距离查询误报 0，故没有采用该接口。每次 Bullet 距离都会和碰撞体包围球给出的下界核对；不自洽则拒绝输出。这不是把 Bullet 动力学结果与 Isaac 动力学结果等同。URDF→USD 转换、接触偏移、形变、线缆和外壳制造误差也不在指标中。最小距离只在评估的离散采样帧内成立，不保证连续时间或实机净距。

当前混训资产 `rough/cts_moe_scratch/thunder/source/doso_train/doso_train/assets.py` 中 `enabled_self_collisions=False`，V24 `code/config.py` 使用该资产，V23 固定案例协议沿用该设置。因此评估里的地面/机身接触或 `undesired_contacts` **不能充当腿与腿、腿与机身的 PhysX 接触记录**；即便输出几何负值，也只是姿态与 URDF 形状推算的重叠。单独的 `pit_task.configure_pit` 有开启自碰撞的代码，但混训的 `mixed_task.configure_mixed` 未调用它，不应以那个专用配置推断本评估已启用自碰撞。

若要测物理接触，应另立无训练更新的配对 Isaac 诊断，在相同源/候选、固定案例和哈希下显式开启自碰撞，并记录接触 link 对、时间和接触力及策略姿态；开启自碰撞会改变轨迹，必须标成新的物理协议。实机归因还需当次故障码、实际加载模型哈希、请求/目标/实际角、关节速度/扭矩及控制映射。几何距离改善本身不构成部署放行。

验证：`ruff check probe.py test_probe.py`；`pytest -q test_probe.py`。测试覆盖实际 URDF 与 manifest 哈希、关节/碰撞体数量、FK 对髋角变化的响应、距离筛选与穷举一致、固定案例配对和非零优化器更新拒绝。
