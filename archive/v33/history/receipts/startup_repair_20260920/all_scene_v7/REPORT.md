# 九地形 V7 固定策略验证

Nine terrains, matched default joint/root states verified across policies. No noise or pushes; fixed10ms delay. 71 cases per policy: 20 all-direction flat starts (6s budget), seven other terrain types at L6/L9 and vx0.3/0.6/1.2, pit0.6/0.9/1.2m at vx0.3/0.6/0.9 (25s budget). No heading feedback. Pit external stop retained. 50Hz trajectories plus per-step 200Hz torque/speed/hard-limit peaks. Fixed cases, not success probabilities over random trials.

|地形|源学生通过|V7学生通过|V7教师通过|V7学生物理失败|
|---|---:|---:|---:|---:|
|平地|14/20|18/20|18/20|0|
|上楼梯|5/6|6/6|6/6|0|
|下楼梯|6/6|6/6|6/6|0|
|上坡|2/6|2/6|0/6|1|
|下坡|4/6|6/6|6/6|0|
|波浪|3/6|5/6|6/6|1|
|粗糙坡|4/6|6/6|6/6|0|
|障碍块|5/6|6/6|6/6|0|
|高台|9/9|0/9|0/9|9|

指标注意：非平地关节姿态偏离包含必要攀爬动作，不能用默认角偏差单独判坏。扭矩/轮速采用每个20ms步内四个5ms子步的峰值；关节时程为50Hz。净空按机身正下方射线测量，在台阶边缘会发生参考地面切换，不能把曲线跳变全部当机身跳动。

这些是固定名义条件的场景覆盖测试，没有覆盖观测噪声、推扰、随机摩擦和全部速度范围；不能当作实机部署验收。本报告比较源策略与清单记录的V7检查点，不推断其他在训策略的表现。
