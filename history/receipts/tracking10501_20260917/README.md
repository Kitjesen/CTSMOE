# Student 10501 tracking evaluation

Four nominal, deterministic student-only rollouts using the same command schedules and terrain cells as student 7501. No observation noise, parameter randomization or external pushes; original actuator delay retained. One trial per case, not training-wide or robustness error. Baseline gains: legs Kp=120 Kd=8; wheels Kp=0 Kd=1.

XY RMSE = sqrt(mean((vx-vx_cmd)^2 + (vy-vy_cmd)^2)). Whole-clip statistics include command transitions and standing segments. Yaw units: rad/s.

| Case | Duration (s) | XY RMSE (m/s) | Previous 7501 XY RMSE | Yaw RMSE (rad/s) | Terminated |
|---|---:|---:|---:|---:|---|
| flat_commands | 22 | 0.05219 | 0.04977 | 0.05476 | False |
| slope_up_L4 | 16 | 0.05569 | 0.04678 | 0.03678 | False |
| stairs_up_L4 | 16 | 0.07977 | 0.13239 | 0.07345 | False |
| stairs_down_L4 | 16 | 0.08706 | 0.09095 | 0.05736 | False |

Flat per-command windows, raw mean velocities, MAE and RMSE are in `flat_comparison.json`; each window excludes at least the first second after switching commands. The final stop window is 21–22 s. Raw complete 50 Hz trajectories and original evaluation config are retained.

Gain provenance: nominal 120/8 retained from DOSO_THUNDER_V4_CFG and explicitly set in the Thunder adapter. Training 0.9–1.1 multiplicative gain randomization is inherited from upstream Go2 events; upstream selected GO2_CFG_UNITREE uses nominal 25/0.5. No gain sweep was performed for this evaluation.
