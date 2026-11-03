# V32 initial yaw continuation
User: disable pure lateral stairs, broaden initial heading and resume training.
Source: V31 STOP checkpoint model_39161.pt, completed39162. Target46194, remaining7032. Verify checkpoint metadata independently on server.
4096 envs; high platform remains0; flat lateral/yaw preserved. Same quotas, reward weights, actuator limits, optimizer states and per-env difficulty.
All stair initial headings:25% aligned;75% uniform[-45,+45] degrees relative inherited outward/backward heading. New755 route groups:straight/back60%, pure side0%, forward+yaw15%, forward+side15%, combined10%. Reference stair groups retain commands but receive same initial heading randomization.
New-route desired pose initialized from actual reset pose. Downgrade compares actual axis progress against half integrated commanded axis displacement rather than unprojected speed*time. Actual platform support geometry unchanged.
CPU8tests passed;ruff passed. Simulator3000zero-update preflight required before any optimizer update, checks seven yaw bins and no side group. Model and dual optimizers must be unchanged. Final assessment remains required, fixed cases are not random success rates.
Frozen V31 preserved. Box40 candidate NOT applied. No reward/LR changes. No hardware release.

Budget correction: initial coordinator169417 inherited10000 in orchestration only. Killed coordinator while preflight169418 remained0updates. Preserved original STATUS.json and continuous_v31.py remotely. Replacement continuous_v32.py waits for same preflight, verifies receipt and7032 remaining before training; authoritative STATUS_v32.json. Never restart original coordinator.

Additional inherited CPU tests:18passed,1failed because legacy test mocks upstream commands while current stair_speed imports local stop_commands (Isaac dependency unavailable locally). Runtime code unchanged; corrected dependency mock in tests/test_stair_speed_compat.py for CPU test.

Corrected CPU stair tests4/4passed; inherited mixed tests15/15passed. Current coordinator169897, preflight169418 at900/3000steps, no training update yet; STATUS_v32.json7032updates to46194. Original training stopped39162. Baseline assessment after preflight, automatic training only after validations.
