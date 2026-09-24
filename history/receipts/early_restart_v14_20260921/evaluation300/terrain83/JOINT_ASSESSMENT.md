# V13逐关节评估

全部数值来自有效回合时程；失败后帧不参与统计。RMS按单个关节沿时间计算。
角度偏差相对默认站姿，不是目标跟踪误差；攀爬阶段偏离默认姿态不自动判错。
4个轮关节单列转速/力矩，不用累计旋转角评价站姿。
CSV包含所有地形、方向、策略、窗口及请求/限位后目标/实际角度。窗口不足会标记，不能当完整通过。

## early_31831：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.037 / 2.1|-0.778 / -0.169 / -0.037|0.022|0.169|
|fr_thigh_joint|0.156 / 8.9|-0.154 / -0.154 / -0.156|0.091|0.240|
|fr_calf_joint|0.086 / 4.9|+0.105 / +0.105 / -0.086|0.030|0.041|
|fl_hip_joint|0.038 / 2.2|+0.392 / +0.169 / +0.038|0.020|0.169|
|fl_thigh_joint|0.111 / 6.4|-0.138 / -0.138 / -0.111|0.069|0.217|
|fl_calf_joint|0.113 / 6.5|-0.204 / -0.204 / -0.113|0.076|0.068|
|rr_hip_joint|0.021 / 1.2|+0.087 / +0.087 / -0.021|0.011|0.170|
|rr_thigh_joint|0.352 / 20.2|+0.324 / +0.324 / +0.352|0.260|0.213|
|rr_calf_joint|0.213 / 12.2|-0.308 / -0.308 / -0.213|0.139|0.078|
|rl_hip_joint|0.021 / 1.2|-0.965 / -0.169 / -0.021|0.010|0.169|
|rl_thigh_joint|0.220 / 12.6|-0.192 / -0.192 / -0.220|0.172|0.142|
|rl_calf_joint|0.199 / 11.4|+0.277 / +0.277 / +0.199|0.163|0.087|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|stairs_down/9/forward|0.211|0.192|True|
|fr_thigh_joint|obstacles/9/forward|0.718|0.690|True|
|fr_calf_joint|wave/6/forward|0.885|1.050|True|
|fl_hip_joint|wave/6/forward|0.323|0.472|True|
|fl_thigh_joint|flat/0/reverse|0.483|0.477|True|
|fl_calf_joint|wave/6/forward|0.620|0.807|True|
|rr_hip_joint|wave/6/forward|0.211|0.169|True|
|rr_thigh_joint|rough_slope/9/forward|0.486|0.479|True|
|rr_calf_joint|wave/6/forward|0.640|0.717|True|
|rl_hip_joint|wave/9/forward|0.211|0.224|True|
|rl_thigh_joint|wave/9/forward|0.538|0.607|True|
|rl_calf_joint|wave/9/forward|0.579|0.887|True|

## early_31831_teacher：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.037 / 2.1|-0.736 / -0.169 / -0.037|0.023|0.169|
|fr_thigh_joint|0.137 / 7.8|-0.146 / -0.146 / -0.137|0.083|0.230|
|fr_calf_joint|0.065 / 3.7|+0.102 / +0.102 / -0.065|0.021|0.084|
|fl_hip_joint|0.042 / 2.4|+0.364 / +0.169 / +0.042|0.018|0.169|
|fl_thigh_joint|0.099 / 5.7|-0.108 / -0.108 / -0.099|0.051|0.225|
|fl_calf_joint|0.119 / 6.8|-0.213 / -0.213 / -0.119|0.077|0.049|
|rr_hip_joint|0.026 / 1.5|+0.115 / +0.115 / -0.026|0.010|0.094|
|rr_thigh_joint|0.298 / 17.1|+0.300 / +0.300 / +0.298|0.248|0.211|
|rr_calf_joint|0.214 / 12.3|-0.302 / -0.302 / -0.214|0.147|0.090|
|rl_hip_joint|0.021 / 1.2|-1.556 / -0.169 / -0.021|0.008|0.169|
|rl_thigh_joint|0.175 / 10.0|-0.162 / -0.162 / -0.175|0.153|0.154|
|rl_calf_joint|0.178 / 10.2|+0.254 / +0.254 / +0.178|0.156|0.064|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/yaw_left|0.249|0.186|True|
|fr_thigh_joint|wave/9/forward|0.538|0.678|True|
|fr_calf_joint|obstacles/9/forward|1.029|1.163|True|
|fl_hip_joint|stairs_down/6/forward|0.210|0.169|True|
|fl_thigh_joint|flat/0/reverse|0.476|0.461|True|
|fl_calf_joint|wave/6/forward|0.668|0.810|True|
|rr_hip_joint|wave/9/forward|0.211|0.480|True|
|rr_thigh_joint|wave/6/forward|0.491|0.846|True|
|rr_calf_joint|wave/9/forward|0.982|0.935|True|
|rl_hip_joint|wave/9/forward|0.210|0.375|True|
|rl_thigh_joint|wave/9/forward|0.494|0.557|True|
|rl_calf_joint|wave/9/forward|0.603|0.696|True|

## V14_32131：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.028 / 1.6|-1.270 / -0.169 / -0.028|0.020|0.169|
|fr_thigh_joint|0.114 / 6.5|+0.077 / +0.077 / +0.114|0.066|0.240|
|fr_calf_joint|0.103 / 5.9|+0.041 / +0.041 / -0.103|0.086|0.038|
|fl_hip_joint|0.044 / 2.5|+0.808 / +0.169 / +0.044|0.040|0.169|
|fl_thigh_joint|0.197 / 11.3|-0.150 / -0.150 / -0.197|0.109|0.240|
|fl_calf_joint|0.138 / 7.9|+0.060 / +0.060 / +0.138|0.123|0.033|
|rr_hip_joint|0.030 / 1.7|+0.552 / +0.169 / +0.030|0.026|0.143|
|rr_thigh_joint|0.219 / 12.5|+0.217 / +0.217 / +0.219|0.177|0.240|
|rr_calf_joint|0.041 / 2.4|-0.049 / -0.049 / +0.041|0.031|0.041|
|rl_hip_joint|0.016 / 0.9|-1.323 / -0.169 / +0.016|0.008|0.169|
|rl_thigh_joint|0.234 / 13.4|-0.233 / -0.233 / -0.234|0.195|0.206|
|rl_calf_joint|0.030 / 1.7|+0.119 / +0.119 / +0.030|0.019|0.092|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|wave/9/forward|0.304|0.503|True|
|fr_thigh_joint|obstacles/9/forward|0.838|0.773|True|
|fr_calf_joint|obstacles/9/forward|0.926|1.011|True|
|fl_hip_joint|wave/6/forward|0.253|0.386|True|
|fl_thigh_joint|wave/9/forward|0.493|0.434|True|
|fl_calf_joint|obstacles/9/forward|0.816|1.023|True|
|rr_hip_joint|wave/9/forward|0.211|0.169|True|
|rr_thigh_joint|slope_down/9/forward|0.519|0.511|True|
|rr_calf_joint|wave/6/forward|0.793|0.734|True|
|rl_hip_joint|wave/9/forward|0.211|0.317|True|
|rl_thigh_joint|wave/9/forward|0.579|0.576|True|
|rl_calf_joint|wave/6/forward|0.660|0.916|True|

## V14_32131_teacher：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.027 / 1.6|-1.207 / -0.169 / -0.027|0.020|0.169|
|fr_thigh_joint|0.101 / 5.8|+0.062 / +0.062 / +0.101|0.059|0.204|
|fr_calf_joint|0.102 / 5.9|+0.006 / +0.006 / -0.102|0.091|0.076|
|fl_hip_joint|0.042 / 2.4|+0.822 / +0.169 / +0.042|0.039|0.169|
|fl_thigh_joint|0.186 / 10.7|-0.136 / -0.136 / -0.186|0.107|0.240|
|fl_calf_joint|0.140 / 8.0|+0.070 / +0.070 / +0.140|0.122|0.044|
|rr_hip_joint|0.033 / 1.9|+0.618 / +0.169 / +0.033|0.026|0.093|
|rr_thigh_joint|0.200 / 11.4|+0.204 / +0.204 / +0.200|0.186|0.229|
|rr_calf_joint|0.055 / 3.1|-0.034 / -0.034 / +0.055|0.036|0.040|
|rl_hip_joint|0.012 / 0.7|-1.370 / -0.169 / +0.012|0.008|0.169|
|rl_thigh_joint|0.222 / 12.7|-0.219 / -0.219 / -0.222|0.191|0.208|
|rl_calf_joint|0.035 / 2.0|+0.067 / +0.067 / -0.035|0.020|0.098|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|wave/9/forward|0.494|0.569|True|
|fr_thigh_joint|wave/9/forward|0.768|0.773|True|
|fr_calf_joint|wave/6/forward|1.314|1.394|True|
|fl_hip_joint|stairs_down/6/forward|0.210|0.169|True|
|fl_thigh_joint|wave/9/forward|0.678|0.635|True|
|fl_calf_joint|obstacles/9/forward|0.881|1.023|True|
|rr_hip_joint|wave/9/forward|0.210|0.423|True|
|rr_thigh_joint|wave/9/forward|0.588|1.142|True|
|rr_calf_joint|wave/6/forward|1.061|1.089|True|
|rl_hip_joint|flat/0/side_turn|0.216|0.311|True|
|rl_thigh_joint|wave/9/forward|0.444|0.451|True|
|rl_calf_joint|wave/6/forward|0.631|0.651|True|
