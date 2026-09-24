# Flat long-run request
Source: completed45036, saved by intentional STOP. Original training exited cleanly after save.
Recording: deterministic student mean, retained training observation noise and randomization, actual atlas flat band. Fixed initial q; 2s zero then28s vx1m/s, vy/yaw0. First termination ends clip. No optimizer update.
Recording-only episode duration32s applied at runtime after training-profile validation; training remains25s. Initial attempt changed config too early and was rejected at zero updates; log/runtime preserved.
Supervisor202004 records then resumes exact1158 remaining updates to46194 and original final assessment. Shared lock held, no parallel GPU. Frozen training code unchanged.

Completed: all1500 control steps/30seconds, no first-episode termination. Model/PPO/student optimizer hashes unchanged. Training resumed PID202797, observed45064. Actual forward25.489m, lateral1.857m, steady vx.9156, twelve-leg startup RMS.1248, final RMS.1126, final height.4296m. One fixed sample, not success rate. CPU replay subtitle originally included wheel angles; rejected video preserved and final subtitle corrected to twelve leg joints before delivery.
