# V13逐关节评估

全部数值来自有效回合时程；失败后帧不参与统计。RMS按单个关节沿时间计算。
角度偏差相对默认站姿，不是目标跟踪误差；攀爬阶段偏离默认姿态不自动判错。
4个轮关节单列转速/力矩，不用累计旋转角评价站姿。
CSV包含所有地形、方向、策略、窗口及请求/限位后目标/实际角度。窗口不足会标记，不能当完整通过。

## early_31831：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.061 / 3.5|-0.772 / -0.169 / -0.061|0.036|0.169|
|fr_thigh_joint|0.213 / 12.2|-0.206 / -0.206 / -0.213|0.104|0.240|
|fr_calf_joint|0.089 / 5.1|+0.108 / +0.108 / -0.089|0.045|0.041|
|fl_hip_joint|0.035 / 2.0|+0.395 / +0.169 / +0.035|0.015|0.169|
|fl_thigh_joint|0.122 / 7.0|+0.157 / +0.157 / +0.122|0.070|0.217|
|fl_calf_joint|0.113 / 6.5|-0.215 / -0.215 / -0.113|0.078|0.068|
|rr_hip_joint|0.021 / 1.2|+0.086 / +0.086 / -0.021|0.010|0.170|
|rr_thigh_joint|0.349 / 20.0|+0.327 / +0.327 / +0.349|0.275|0.213|
|rr_calf_joint|0.225 / 12.9|-0.312 / -0.312 / -0.225|0.136|0.082|
|rl_hip_joint|0.023 / 1.3|-0.963 / -0.169 / -0.023|0.012|0.169|
|rl_thigh_joint|0.214 / 12.3|-0.195 / -0.195 / -0.214|0.171|0.142|
|rl_calf_joint|0.191 / 11.0|+0.268 / +0.268 / +0.191|0.149|0.087|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/left|0.209|0.169|True|
|fr_thigh_joint|flat/0/diagonal|0.418|0.433|True|
|fr_calf_joint|flat/0/left|0.449|0.571|True|
|fl_hip_joint|flat/0/left|0.210|0.169|True|
|fl_thigh_joint|flat/0/reverse|0.466|0.453|True|
|fl_calf_joint|flat/0/right|0.510|0.570|True|
|rr_hip_joint|flat/0/left|0.210|0.169|True|
|rr_thigh_joint|flat/0/reverse|0.414|0.447|True|
|rr_calf_joint|flat/0/diagonal|0.402|0.457|True|
|rl_hip_joint|flat/0/yaw_right|0.210|0.169|True|
|rl_thigh_joint|flat/0/yaw_left|0.281|0.257|True|
|rl_calf_joint|flat/0/side_turn|0.474|0.539|True|

## early_31831_teacher：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.051 / 2.9|-0.675 / -0.169 / -0.051|0.032|0.169|
|fr_thigh_joint|0.170 / 9.7|-0.164 / -0.164 / -0.170|0.096|0.230|
|fr_calf_joint|0.068 / 3.9|+0.111 / +0.111 / -0.068|0.028|0.084|
|fl_hip_joint|0.042 / 2.4|+0.360 / +0.169 / +0.042|0.017|0.169|
|fl_thigh_joint|0.108 / 6.2|-0.132 / -0.132 / -0.108|0.053|0.225|
|fl_calf_joint|0.085 / 4.9|-0.170 / -0.170 / -0.085|0.062|0.039|
|rr_hip_joint|0.027 / 1.5|+0.107 / +0.107 / -0.027|0.010|0.094|
|rr_thigh_joint|0.334 / 19.1|+0.322 / +0.322 / +0.334|0.282|0.211|
|rr_calf_joint|0.225 / 12.9|-0.312 / -0.312 / -0.225|0.152|0.090|
|rl_hip_joint|0.022 / 1.3|-1.559 / -0.169 / -0.022|0.011|0.169|
|rl_thigh_joint|0.193 / 11.0|-0.160 / -0.160 / -0.193|0.160|0.154|
|rl_calf_joint|0.178 / 10.2|+0.255 / +0.255 / +0.178|0.145|0.064|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/yaw_left|0.268|0.184|True|
|fr_thigh_joint|flat/0/diagonal|0.393|0.360|True|
|fr_calf_joint|flat/0/left|0.436|0.540|True|
|fl_hip_joint|flat/0/forward_turn|0.210|0.169|True|
|fl_thigh_joint|flat/0/reverse|0.496|0.537|True|
|fl_calf_joint|flat/0/diagonal_back|0.519|0.572|True|
|rr_hip_joint|flat/0/right|0.210|0.189|True|
|rr_thigh_joint|flat/0/reverse|0.361|0.475|True|
|rr_calf_joint|flat/0/right|0.493|0.629|True|
|rl_hip_joint|flat/0/right|0.207|0.169|True|
|rl_thigh_joint|flat/0/reverse|0.212|0.247|True|
|rl_calf_joint|flat/0/side_turn|0.357|0.501|True|

## V14_32131：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.047 / 2.7|-1.205 / -0.169 / -0.047|0.030|0.169|
|fr_thigh_joint|0.116 / 6.7|-0.103 / -0.103 / -0.116|0.069|0.240|
|fr_calf_joint|0.129 / 7.4|+0.012 / +0.012 / -0.129|0.108|0.038|
|fl_hip_joint|0.044 / 2.5|+0.813 / +0.169 / +0.044|0.035|0.169|
|fl_thigh_joint|0.182 / 10.4|-0.157 / -0.157 / -0.182|0.095|0.240|
|fl_calf_joint|0.159 / 9.1|+0.062 / +0.062 / +0.159|0.129|0.033|
|rr_hip_joint|0.037 / 2.1|+0.612 / +0.169 / +0.037|0.023|0.143|
|rr_thigh_joint|0.221 / 12.7|+0.202 / +0.202 / +0.221|0.202|0.240|
|rr_calf_joint|0.107 / 6.1|-0.031 / -0.031 / +0.107|0.049|0.038|
|rl_hip_joint|0.024 / 1.4|-1.684 / -0.169 / -0.024|0.010|0.169|
|rl_thigh_joint|0.240 / 13.8|-0.220 / -0.220 / -0.240|0.206|0.206|
|rl_calf_joint|0.066 / 3.8|+0.094 / +0.094 / -0.066|0.028|0.092|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/left|0.222|0.273|True|
|fr_thigh_joint|flat/0/right|0.230|0.204|True|
|fr_calf_joint|flat/0/left|0.365|0.497|True|
|fl_hip_joint|flat/0/side_turn|0.209|0.169|True|
|fl_thigh_joint|flat/0/side_turn|0.315|0.249|True|
|fl_calf_joint|flat/0/right|0.267|0.376|True|
|rr_hip_joint|flat/0/side_turn|0.210|0.240|True|
|rr_thigh_joint|flat/0/diagonal|0.278|0.318|True|
|rr_calf_joint|flat/0/diagonal_back|0.321|0.434|True|
|rl_hip_joint|flat/0/side_turn|0.217|0.326|True|
|rl_thigh_joint|flat/0/reverse|0.259|0.417|True|
|rl_calf_joint|flat/0/side_turn|0.257|0.422|True|

## V14_32131_teacher：平地前进直接起步首秒

|关节|实际峰值rad / °|请求/目标/实际（实际峰值时）rad|单关节RMS rad|最大目标跳变rad|
|---|---:|---:|---:|---:|
|fr_hip_joint|0.039 / 2.2|-1.148 / -0.169 / -0.039|0.027|0.169|
|fr_thigh_joint|0.113 / 6.5|+0.079 / +0.079 / +0.113|0.072|0.204|
|fr_calf_joint|0.118 / 6.7|+0.023 / +0.023 / -0.118|0.104|0.078|
|fl_hip_joint|0.046 / 2.6|+0.835 / +0.169 / +0.046|0.036|0.169|
|fl_thigh_joint|0.202 / 11.6|-0.154 / -0.154 / -0.202|0.105|0.240|
|fl_calf_joint|0.155 / 8.9|+0.085 / +0.085 / +0.155|0.127|0.043|
|rr_hip_joint|0.051 / 2.9|+0.688 / +0.169 / +0.051|0.029|0.093|
|rr_thigh_joint|0.268 / 15.4|+0.243 / +0.243 / +0.268|0.217|0.229|
|rr_calf_joint|0.109 / 6.3|-0.080 / -0.080 / +0.109|0.059|0.044|
|rl_hip_joint|0.022 / 1.2|-1.795 / -0.169 / -0.022|0.009|0.169|
|rl_thigh_joint|0.252 / 14.5|-0.235 / -0.235 / -0.252|0.218|0.208|
|rl_calf_joint|0.071 / 4.1|+0.139 / +0.139 / -0.071|0.037|0.098|

各关节跨地形最差首秒固定案例（按实际绝对偏移排序，不等于动作不合理）：

|关节|地形/等级/方向|实际峰值rad|目标峰值rad|完整首秒|
|---|---|---:|---:|---|
|fr_hip_joint|flat/0/yaw_left|0.233|0.205|True|
|fr_thigh_joint|flat/0/right|0.262|0.253|True|
|fr_calf_joint|flat/0/left|0.335|0.471|True|
|fl_hip_joint|flat/0/yaw_left|0.207|0.169|True|
|fl_thigh_joint|flat/0/side_turn|0.302|0.223|True|
|fl_calf_joint|flat/0/diagonal|0.373|0.503|True|
|rr_hip_joint|flat/0/left|0.210|0.169|True|
|rr_thigh_joint|flat/0/forward|0.268|0.291|True|
|rr_calf_joint|flat/0/right|0.322|0.436|True|
|rl_hip_joint|flat/0/right|0.209|0.169|True|
|rl_thigh_joint|flat/0/reverse|0.280|0.251|True|
|rl_calf_joint|flat/0/right|0.279|0.422|True|
