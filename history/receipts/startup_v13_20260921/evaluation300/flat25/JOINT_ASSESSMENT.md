# V13逐关节评估

全部数值来自有效回合时程；失败后帧不参与统计。RMS按单个关节沿时间计算。
角度偏差相对默认站姿，不是目标跟踪误差；攀爬阶段偏离默认姿态不自动判错。
4个轮关节单列转速/力矩，不用累计旋转角评价站姿。
CSV包含所有地形、方向、策略、窗口及请求/限位后目标/实际角度。窗口不足会标记，不能当完整通过。

## V9_44711：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.116 / 6.6|-0.101 / -0.101 / -0.116|0.085|0.169|
|fr_thigh_joint|0.809 / 46.3|+0.477 / +0.477 / +0.809|0.607|0.240|
|fr_calf_joint|0.324 / 18.5|-0.182 / -0.182 / -0.324|0.211|0.160|
|fl_hip_joint|0.610 / 35.0|-0.458 / -0.458 / -0.610|0.554|0.240|
|fl_thigh_joint|0.560 / 32.1|+0.463 / +0.463 / +0.560|0.498|0.131|
|fl_calf_joint|0.904 / 51.8|-1.111 / -1.023 / -0.904|0.797|0.209|
|rr_hip_joint|0.181 / 10.4|-0.085 / -0.085 / -0.181|0.091|0.150|
|rr_thigh_joint|0.288 / 16.5|+0.104 / +0.104 / +0.288|0.169|0.240|
|rr_calf_joint|0.191 / 11.0|-0.252 / -0.252 / -0.191|0.120|0.170|
|rl_hip_joint|0.170 / 9.7|-0.348 / -0.169 / -0.170|0.118|0.169|
|rl_thigh_joint|0.290 / 16.6|+0.339 / +0.339 / +0.290|0.199|0.240|
|rl_calf_joint|0.125 / 7.2|+0.204 / +0.204 / +0.125|0.076|0.118|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/diagonal|0.155|0.169|True|
|fr_thigh_joint|flat/0/forward|0.809|0.628|True|
|fr_calf_joint|flat/0/forward|0.324|0.389|True|
|fl_hip_joint|flat/0/forward|0.610|0.569|True|
|fl_thigh_joint|flat/0/forward|0.560|0.573|True|
|fl_calf_joint|flat/0/forward|0.904|1.023|True|
|rr_hip_joint|flat/0/forward|0.181|0.314|True|
|rr_thigh_joint|flat/0/forward|0.288|0.443|True|
|rr_calf_joint|flat/0/yaw_right|0.255|0.386|True|
|rl_hip_joint|flat/0/forward|0.170|0.169|True|
|rl_thigh_joint|flat/0/forward|0.290|0.470|True|
|rl_calf_joint|flat/0/diagonal|0.256|0.321|True|

## V9_44711_teacher：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.105 / 6.0|-0.158 / -0.158 / -0.105|0.080|0.169|
|fr_thigh_joint|0.790 / 45.2|+0.526 / +0.526 / +0.790|0.611|0.240|
|fr_calf_joint|0.294 / 16.9|-0.189 / -0.189 / -0.294|0.198|0.158|
|fl_hip_joint|0.610 / 35.0|-0.413 / -0.413 / -0.610|0.549|0.240|
|fl_thigh_joint|0.548 / 31.4|+0.465 / +0.465 / +0.548|0.485|0.114|
|fl_calf_joint|0.933 / 53.5|-1.015 / -1.015 / -0.933|0.820|0.201|
|rr_hip_joint|0.181 / 10.4|-0.120 / -0.120 / -0.181|0.096|0.165|
|rr_thigh_joint|0.299 / 17.1|+0.136 / +0.136 / +0.299|0.179|0.240|
|rr_calf_joint|0.198 / 11.3|-0.281 / -0.281 / -0.198|0.113|0.240|
|rl_hip_joint|0.153 / 8.8|-0.349 / -0.169 / -0.153|0.108|0.169|
|rl_thigh_joint|0.268 / 15.4|+0.326 / +0.326 / +0.268|0.187|0.240|
|rl_calf_joint|0.131 / 7.5|+0.210 / +0.210 / +0.131|0.077|0.148|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/left|0.149|0.169|True|
|fr_thigh_joint|flat/0/forward|0.790|0.629|True|
|fr_calf_joint|flat/0/diagonal_back|0.319|0.480|True|
|fl_hip_joint|flat/0/forward|0.610|0.569|True|
|fl_thigh_joint|flat/0/forward|0.548|0.578|True|
|fl_calf_joint|flat/0/forward|0.962|1.023|True|
|rr_hip_joint|flat/0/forward|0.181|0.319|True|
|rr_thigh_joint|flat/0/forward|0.299|0.436|True|
|rr_calf_joint|flat/0/forward|0.284|0.382|True|
|rl_hip_joint|flat/0/yaw_right|0.170|0.169|True|
|rl_thigh_joint|flat/0/forward|0.268|0.480|True|
|rl_calf_joint|flat/0/yaw_left|0.214|0.382|True|

## V6_43202：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.058 / 3.3|-0.197 / -0.169 / -0.058|0.031|0.169|
|fr_thigh_joint|0.409 / 23.4|+0.203 / +0.203 / +0.409|0.330|0.240|
|fr_calf_joint|0.151 / 8.7|+0.001 / +0.001 / -0.151|0.102|0.165|
|fl_hip_joint|0.111 / 6.4|-0.047 / -0.047 / -0.111|0.038|0.240|
|fl_thigh_joint|0.152 / 8.7|-0.093 / -0.093 / -0.152|0.097|0.070|
|fl_calf_joint|0.072 / 4.1|-0.044 / -0.044 / -0.072|0.027|0.048|
|rr_hip_joint|0.077 / 4.4|-0.000 / -0.000 / +0.077|0.042|0.174|
|rr_thigh_joint|0.235 / 13.5|+0.209 / +0.209 / +0.235|0.129|0.231|
|rr_calf_joint|0.145 / 8.3|+0.006 / +0.006 / +0.145|0.109|0.120|
|rl_hip_joint|0.044 / 2.5|-0.331 / -0.169 / +0.044|0.014|0.169|
|rl_thigh_joint|0.387 / 22.2|-0.371 / -0.371 / -0.387|0.220|0.240|
|rl_calf_joint|0.172 / 9.8|+0.283 / +0.283 / +0.172|0.147|0.210|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/yaw_right|0.210|0.169|True|
|fr_thigh_joint|flat/0/forward|0.409|0.516|True|
|fr_calf_joint|flat/0/right|0.353|0.443|True|
|fl_hip_joint|flat/0/yaw_right|0.286|0.250|True|
|fl_thigh_joint|flat/0/diagonal_back|0.289|0.313|True|
|fl_calf_joint|flat/0/diagonal_back|0.299|0.415|True|
|rr_hip_joint|flat/0/side_turn|0.197|0.169|True|
|rr_thigh_joint|flat/0/yaw_right|0.259|0.363|True|
|rr_calf_joint|flat/0/right|0.323|0.494|True|
|rl_hip_joint|flat/0/yaw_right|0.206|0.169|True|
|rl_thigh_joint|flat/0/forward|0.387|0.610|True|
|rl_calf_joint|flat/0/yaw_left|0.278|0.411|True|

## V13_45011：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.119 / 6.8|-0.202 / -0.169 / -0.119|0.084|0.169|
|fr_thigh_joint|0.489 / 28.0|+0.311 / +0.311 / +0.489|0.365|0.240|
|fr_calf_joint|0.235 / 13.5|-0.107 / -0.107 / -0.235|0.164|0.172|
|fl_hip_joint|0.409 / 23.4|-0.250 / -0.250 / -0.409|0.242|0.240|
|fl_thigh_joint|0.106 / 6.1|-0.078 / -0.078 / -0.106|0.079|0.127|
|fl_calf_joint|0.263 / 15.0|-0.304 / -0.304 / -0.263|0.194|0.092|
|rr_hip_joint|0.244 / 14.0|-0.166 / -0.166 / -0.244|0.153|0.240|
|rr_thigh_joint|0.508 / 29.1|+0.380 / +0.380 / +0.508|0.243|0.238|
|rr_calf_joint|0.143 / 8.2|-0.294 / -0.294 / -0.143|0.079|0.073|
|rl_hip_joint|0.063 / 3.6|-0.121 / -0.121 / -0.063|0.036|0.169|
|rl_thigh_joint|0.556 / 31.9|-0.506 / -0.506 / -0.556|0.311|0.240|
|rl_calf_joint|0.220 / 12.6|+0.373 / +0.373 / +0.220|0.120|0.238|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/yaw_right|0.199|0.169|True|
|fr_thigh_joint|flat/0/forward|0.489|0.485|True|
|fr_calf_joint|flat/0/diagonal_back|0.402|0.540|True|
|fl_hip_joint|flat/0/forward|0.409|0.511|True|
|fl_thigh_joint|flat/0/diagonal_back|0.313|0.305|True|
|fl_calf_joint|flat/0/diagonal_back|0.308|0.394|True|
|rr_hip_joint|flat/0/forward|0.244|0.351|True|
|rr_thigh_joint|flat/0/forward|0.508|0.542|True|
|rr_calf_joint|flat/0/diagonal_back|0.254|0.422|True|
|rl_hip_joint|flat/0/diagonal_back|0.210|0.169|True|
|rl_thigh_joint|flat/0/forward|0.556|0.602|True|
|rl_calf_joint|flat/0/yaw_left|0.268|0.430|True|

## V13_45011_teacher：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.119 / 6.8|-0.303 / -0.169 / +0.119|0.050|0.231|
|fr_thigh_joint|0.556 / 31.9|+0.537 / +0.537 / +0.556|0.314|0.240|
|fr_calf_joint|0.162 / 9.3|+0.338 / +0.338 / +0.162|0.110|0.240|
|fl_hip_joint|0.610 / 34.9|-0.471 / -0.471 / -0.610|0.478|0.240|
|fl_thigh_joint|0.329 / 18.8|+0.197 / +0.197 / +0.329|0.139|0.158|
|fl_calf_joint|0.844 / 48.4|-0.848 / -0.848 / -0.844|0.625|0.150|
|rr_hip_joint|0.370 / 21.2|-0.151 / -0.151 / -0.370|0.189|0.219|
|rr_thigh_joint|0.553 / 31.7|+0.356 / +0.356 / +0.553|0.361|0.240|
|rr_calf_joint|0.574 / 32.9|-0.659 / -0.659 / -0.574|0.282|0.178|
|rl_hip_joint|0.210 / 12.0|-0.602 / -0.169 / -0.210|0.133|0.169|
|rl_thigh_joint|0.536 / 30.7|-0.464 / -0.464 / -0.536|0.322|0.240|
|rl_calf_joint|0.351 / 20.1|-0.132 / -0.132 / -0.351|0.236|0.240|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/left|0.204|0.169|True|
|fr_thigh_joint|flat/0/forward|0.556|0.537|True|
|fr_calf_joint|flat/0/diagonal_back|0.396|0.556|True|
|fl_hip_joint|flat/0/forward|0.610|0.569|True|
|fl_thigh_joint|flat/0/forward|0.329|0.315|True|
|fl_calf_joint|flat/0/forward|0.844|0.937|True|
|rr_hip_joint|flat/0/forward|0.370|0.523|True|
|rr_thigh_joint|flat/0/forward|0.553|0.543|True|
|rr_calf_joint|flat/0/forward|0.574|0.777|True|
|rl_hip_joint|flat/0/diagonal_back|0.210|0.169|True|
|rl_thigh_joint|flat/0/forward|0.536|0.569|True|
|rl_calf_joint|flat/0/forward|0.351|0.453|True|
