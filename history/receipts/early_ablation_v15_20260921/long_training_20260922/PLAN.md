# V15 continuation after corrected evaluation

User authorization: the user requested 10,000 training updates, and challenged stopping after only 300. Corrected paired evidence supports a bounded continuation. This is learning authorization, not deployment acceptance.

Source: V15 completed 32,131, `/root/cts-early-ablation-v15-20260921/runs/pilot300/model_32130.pt`. Preserve its model and both optimizers. The 300 updates already consumed count toward the total 10,000 from source 31,831. Continue 9,700: 700 + 2,000 + 2,000 + 2,000 + 3,000, ending at completed 41,831.

Runtime: unchanged frozen V15 training code, 4,096 environments, same rewards, reset handling, actuator limits, command sampling and terrain distribution. No old pipeline is restarted. Data output: `/root/autodl-tmp/cts-v15-long-10000-20260922`.

Each stage has 332 nine-terrain and 80 full-episode flat fixed paired tests using the corrected evaluator. Single-case regressions are observations, not automatic training rejection. Keep model checkpoints and all evaluations. These fixed cases are not random success rates.

Stop on process/configuration/checkpoint/evaluation failures, NaN/Inf, the existing applied-torque guard, or disk reserve below 10 GiB. During training the disk check runs every 10 seconds and requests the existing STOP save mechanism. Two consecutive evaluations below 10/20 in the same flat policy role stop for review. This is a declared experiment-management threshold, not a statistical confidence or final acceptance criterion.

Shared nonblocking flock prevents duplicate GPU training. Original source hashes, all frozen V15 source hashes, and corrected evaluator hashes are checked throughout. No automatic retry or overwrite of an existing run. Finishing the budget does not mark the final policy accepted and does not deploy hardware.

Verified start: remote PID 8285, 4096 environments, completed 32137 observed after resuming 32131. Model, PPO optimizer, student optimizer, terrain assignment/membership/allocation and exact terrain restoration all verified by the run receipt. Latest iteration 6.759 s. The first stage target is 32831; the final budget target remains 41831.
