# V13逐关节评估

全部数值来自有效回合时程；失败后帧不参与统计。RMS按单个关节沿时间计算。
角度偏差相对默认站姿，不是目标跟踪误差；攀爬阶段偏离默认姿态不自动判错。
4个轮关节单列转速/力矩，不用累计旋转角评价站姿。
CSV包含所有地形、方向、策略、窗口及请求/限位后目标/实际角度。窗口不足会标记，不能当完整通过。

## V9_44711：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.116 / 6.6|-0.097 / -0.097 / -0.116|0.087|0.169|
|fr_thigh_joint|0.812 / 46.5|+0.461 / +0.461 / +0.812|0.600|0.240|
|fr_calf_joint|0.266 / 15.2|-0.172 / -0.172 / -0.266|0.195|0.160|
|fl_hip_joint|0.610 / 35.0|-0.447 / -0.447 / -0.610|0.555|0.240|
|fl_thigh_joint|0.565 / 32.4|+0.481 / +0.481 / +0.565|0.503|0.129|
|fl_calf_joint|0.907 / 52.0|-0.994 / -0.994 / -0.907|0.803|0.210|
|rr_hip_joint|0.183 / 10.5|-0.087 / -0.087 / -0.183|0.092|0.150|
|rr_thigh_joint|0.286 / 16.4|+0.106 / +0.106 / +0.286|0.159|0.240|
|rr_calf_joint|0.223 / 12.8|-0.269 / -0.269 / -0.223|0.146|0.165|
|rl_hip_joint|0.178 / 10.2|-0.346 / -0.169 / -0.178|0.120|0.169|
|rl_thigh_joint|0.289 / 16.6|+0.347 / +0.347 / +0.289|0.191|0.240|
|rl_calf_joint|0.123 / 7.0|+0.203 / +0.203 / +0.123|0.074|0.118|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|wave/6/forward|0.210|0.272|True|
|fr_thigh_joint|pit/6/forward|0.857|0.688|True|
|fr_calf_joint|wave/6/forward|0.465|0.396|True|
|fl_hip_joint|rough_slope/9/forward|0.610|0.569|True|
|fl_thigh_joint|pit/9/forward|0.581|0.649|True|
|fl_calf_joint|flat/0/forward|0.953|1.023|True|
|rr_hip_joint|wave/6/forward|0.241|0.418|True|
|rr_thigh_joint|wave/6/forward|0.484|0.418|True|
|rr_calf_joint|wave/9/forward|0.601|0.617|True|
|rl_hip_joint|wave/9/forward|0.211|0.539|True|
|rl_thigh_joint|wave/6/forward|0.526|0.637|True|
|rl_calf_joint|wave/9/forward|0.466|0.570|True|

## V9_44711_teacher：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.108 / 6.2|-0.180 / -0.169 / -0.108|0.084|0.169|
|fr_thigh_joint|0.791 / 45.3|+0.508 / +0.508 / +0.791|0.608|0.240|
|fr_calf_joint|0.239 / 13.7|-0.171 / -0.171 / -0.239|0.182|0.158|
|fl_hip_joint|0.610 / 35.0|-0.440 / -0.440 / -0.610|0.550|0.240|
|fl_thigh_joint|0.547 / 31.3|+0.454 / +0.454 / +0.547|0.482|0.115|
|fl_calf_joint|0.947 / 54.3|-1.043 / -1.023 / -0.947|0.832|0.198|
|rr_hip_joint|0.180 / 10.3|-0.116 / -0.116 / -0.180|0.101|0.166|
|rr_thigh_joint|0.297 / 17.0|+0.122 / +0.122 / +0.297|0.164|0.240|
|rr_calf_joint|0.226 / 12.9|-0.302 / -0.302 / -0.226|0.143|0.240|
|rl_hip_joint|0.159 / 9.1|-0.347 / -0.169 / -0.159|0.108|0.169|
|rl_thigh_joint|0.268 / 15.3|+0.333 / +0.333 / +0.268|0.182|0.240|
|rl_calf_joint|0.130 / 7.4|+0.212 / +0.212 / +0.130|0.071|0.148|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|wave/6/forward|0.210|0.169|True|
|fr_thigh_joint|rough_slope/9/forward|0.894|0.713|True|
|fr_calf_joint|stairs_up/9/forward|0.852|1.081|True|
|fl_hip_joint|pit/4/forward|0.610|0.569|True|
|fl_thigh_joint|wave/9/forward|0.833|0.773|True|
|fl_calf_joint|flat/0/forward|0.969|1.023|True|
|rr_hip_joint|wave/6/forward|0.210|0.169|True|
|rr_thigh_joint|wave/9/forward|0.593|0.615|True|
|rr_calf_joint|wave/6/forward|0.745|0.754|True|
|rl_hip_joint|wave/6/forward|0.246|0.545|True|
|rl_thigh_joint|wave/9/forward|0.841|0.885|True|
|rl_calf_joint|wave/9/forward|0.608|0.704|True|

## V13_45011：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.099 / 5.6|-0.131 / -0.131 / -0.099|0.074|0.169|
|fr_thigh_joint|0.463 / 26.5|+0.301 / +0.301 / +0.463|0.340|0.240|
|fr_calf_joint|0.201 / 11.5|-0.087 / -0.087 / -0.201|0.151|0.172|
|fl_hip_joint|0.284 / 16.3|-0.207 / -0.207 / -0.284|0.168|0.240|
|fl_thigh_joint|0.105 / 6.0|-0.084 / -0.084 / -0.105|0.062|0.127|
|fl_calf_joint|0.163 / 9.3|-0.206 / -0.206 / -0.163|0.110|0.092|
|rr_hip_joint|0.103 / 5.9|-0.130 / -0.130 / -0.103|0.054|0.213|
|rr_thigh_joint|0.461 / 26.4|+0.343 / +0.343 / +0.461|0.230|0.238|
|rr_calf_joint|0.074 / 4.2|-0.051 / -0.051 / -0.074|0.031|0.067|
|rl_hip_joint|0.049 / 2.8|-0.131 / -0.131 / -0.049|0.030|0.169|
|rl_thigh_joint|0.485 / 27.8|-0.496 / -0.496 / -0.485|0.251|0.240|
|rl_calf_joint|0.189 / 10.8|+0.332 / +0.332 / +0.189|0.097|0.238|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|wave/6/forward|0.416|0.569|True|
|fr_thigh_joint|wave/6/forward|0.930|0.773|True|
|fr_calf_joint|wave/6/forward|0.635|0.716|True|
|fl_hip_joint|rough_slope/6/forward|0.610|0.569|True|
|fl_thigh_joint|wave/6/forward|0.605|0.675|True|
|fl_calf_joint|pit/0/forward|1.060|1.023|True|
|rr_hip_joint|wave/6/forward|0.567|0.569|True|
|rr_thigh_joint|wave/9/forward|0.705|0.773|True|
|rr_calf_joint|wave/9/forward|0.911|0.981|True|
|rl_hip_joint|wave/9/forward|0.212|0.169|True|
|rl_thigh_joint|rough_slope/9/forward|0.766|0.812|True|
|rl_calf_joint|wave/6/forward|0.830|1.069|True|

## V13_45011_teacher：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.114 / 6.5|-0.293 / -0.169 / +0.114|0.057|0.240|
|fr_thigh_joint|0.553 / 31.7|+0.546 / +0.546 / +0.553|0.311|0.240|
|fr_calf_joint|0.176 / 10.1|-0.035 / -0.035 / -0.176|0.115|0.240|
|fl_hip_joint|0.610 / 35.0|-0.481 / -0.481 / -0.610|0.478|0.240|
|fl_thigh_joint|0.334 / 19.1|+0.201 / +0.201 / +0.334|0.140|0.158|
|fl_calf_joint|0.810 / 46.4|-0.867 / -0.867 / -0.810|0.569|0.156|
|rr_hip_joint|0.375 / 21.5|-0.166 / -0.166 / -0.375|0.206|0.219|
|rr_thigh_joint|0.559 / 32.0|+0.363 / +0.363 / +0.559|0.389|0.240|
|rr_calf_joint|0.550 / 31.5|-0.691 / -0.691 / -0.550|0.265|0.176|
|rl_hip_joint|0.210 / 12.0|-0.390 / -0.169 / -0.210|0.154|0.169|
|rl_thigh_joint|0.540 / 31.0|-0.467 / -0.467 / -0.540|0.314|0.240|
|rl_calf_joint|0.295 / 16.9|-0.107 / -0.107 / -0.295|0.191|0.240|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|pit/9/forward|0.328|0.522|True|
|fr_thigh_joint|wave/9/forward|1.200|1.297|True|
|fr_calf_joint|stairs_up/9/forward|0.927|1.144|True|
|fl_hip_joint|flat/0/forward|0.610|0.569|True|
|fl_thigh_joint|wave/9/forward|0.930|0.641|True|
|fl_calf_joint|wave/9/forward|1.080|1.023|True|
|rr_hip_joint|wave/9/forward|0.569|0.569|True|
|rr_thigh_joint|wave/9/forward|0.751|0.831|True|
|rr_calf_joint|wave/9/forward|1.048|1.293|True|
|rl_hip_joint|wave/6/forward|0.322|0.418|True|
|rl_thigh_joint|rough_slope/9/forward|0.725|0.744|True|
|rl_calf_joint|wave/9/forward|1.007|1.235|True|
