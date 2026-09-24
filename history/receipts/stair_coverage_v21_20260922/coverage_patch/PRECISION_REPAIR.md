# Attempt02: preserve the actual mesh cap level

真实 attempt01 样本为 `stairs_down`，8227 个顶点、16200 个三角面。捕获文件没有 row/column，因此不据高度推断行列。原始文件复制为 `fixtures/geometry_failure_attempt01.npz`，SHA256 为 `6cafa384e17d1f2d1c37cd040906e6e3c9d5ab5d3b5ef7c0e606c0d90f2a3ba0`。

失败原因已经由该真实网格确定：band 众数经过 `round(z, 5)` 成为 0.24；网格对应的唯一原始值实际为 0.23999999463558197。旧的 `maximum(original_z, 0.24)` 改写了 107 个原本同层的内区顶点，差值为 5.364418020903372e-9 m。它们的半径是 2.6999998092651367 m，严格落在 r<2.7 的保护区域，因此原来的精确保护正确报错。

修复只改变 cap 的数值来源：仍按原来的 rounded 众数组选择台阶层，但要求该组的原始 mesh Z 只有一个不同值，直接采用该原始值；若同一组有多个不同原始值则拒绝，不平均，不引入容差。奖励、命令、限制和课程均未修改。

全部严格内区顶点和三角坐标哈希检查保留。另对每一个被改变的三角面证明：存在一个 XY 轴，其三个顶点全部位于保护正方形同一侧的外部。因三角面是顶点的凸组合，整个面均在该外半空间内；这个证明也覆盖了三个顶点不在内区但三角面穿过内区的情况。

真实样本修复后：4542 个顶点、9000 个三角面改变；所有变化面距保护正方形至少有一个轴间隔 0.3 m，它们的顶点最小半径为 3.0 m；严格内区顶点、实际输出内区三角顶点及三角坐标前后哈希完全相同。完整证据在 `PRECISION_REPAIR_ACTUAL.json`。

回归测试涵盖该实际样本、真实内区高度被改时仍拒绝、无内区顶点但穿过内区的变化面仍拒绝、众数组含不同原始高度时拒绝。另做了 60 份 pinned heightfield 源实现的 CPU 试算（上下楼 × 10 等级 × 0/0.5/0.999 行内分数），内区精确比较和整面外部证明全部成立；这些试算不能替代远端实际 44 个 selected cell 的 attempt02 收据。

`runtime_prepare/runtime_contract.py` 已程序生成并更新已审 coverage SHA 为 `54788c3c2179de8f9afb62fd0915b3c35ffa7a40691d754cb9a4187ecefa442a`。资格校验新增要求：`cap_original_level_count == 1`、`changed_faces_outside_inner_square == true`、`changed_faces_inner_square_overlap_count == 0`、变动面数量为正且最小轴间隔非负。原 attempt01 无此证明的收据不会被当作 attempt02 合格记录。

没有覆盖远端 attempt01，没有连接 SSH/GPU，没有优化器更新。下一步由主任务部署 payload02 并运行零更新预飞，核验所有实际 selected mesh；此文不宣称 attempt02 已通过。
