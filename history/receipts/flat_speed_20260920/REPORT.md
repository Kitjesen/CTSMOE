# Flat forward/reverse speed expansion

Official commands.py checked on 2026-09-20: initial vx +/-0.5 m/s, update 20000 +/-1, update 50000 +/-2; flat cap +/-2. Limit-command sampling probability 0.2 selects interval endpoints. This is a command range, not proof of achieved speed.

Implemented flat-only vx +/-1.5 to +/-2 linearly over 2000 updates, using absolute saved simulation steps. Other axes and upstream random, limit and zero-command sampling remain in place. The global max_lin_vel used by upstream zero-command time budgeting is unchanged (conservative for faster flat traversal). MixedCommand refresh reapplies the override after terrain membership or upstream curriculum changes. Course state is checkpointed and effective/sampled flat ranges are logged.

Validation: 13 tests passed (flat speed, pit individual, allocation); ruff passed locally. Remote ruff unavailable. Real 4096-env smoke 37001–37009 and inheritance 37009–37011 completed. Course start persisted and limit continued to increase after resume. These isolated validation branches were not used as the formal resume source.

Production saved and stopped at completed update 37283 (model_37282.pt). Resumed from that exact checkpoint into runs/mixed9_4096_60000_flat_speed_20260920, PID 158275, code_flat_speed_20260920_v1, log flat_speed_formal.log, pidfile flat_speed_formal.pid. Original run/code/checkpoint retained. Target 60000, additional updates 22717.

Verified live at update 37288, 6.81 seconds/update: all model/optimizer/terrain/pit/stair/allocation inheritance checks true; 123 flat environments; effective vx +/-1.50125 m/s; actual commands included -1.5 and +1.50007. No applied-torque violation samples in the new run so far. Target +/-2 is reached at approximately completed update 39283. This time-based range expansion does not certify tracking performance at the new speeds.

Current-run verification: remote verify_flat_speed.py, local verified.json. The previous inspect_individual_running.py points to the stopped predecessor and must not be used for current status.
