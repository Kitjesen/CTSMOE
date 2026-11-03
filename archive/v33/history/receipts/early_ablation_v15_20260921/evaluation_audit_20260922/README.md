# V15 evaluation audit (2026-09-22)

Run from this directory after the parent V15 runtime is available:

```powershell
python evaluate_corrected.py --root <workspace> --output <output> `
  --selection <selection.json> --protocol <protocol.json> --code <code_train> `
  --suite terrain83
```

The corrected paired evaluation was started on the authorized remote server in
`/root/autodl-tmp/cts-v15-evaluation-audit-20260922`. It performs zero optimizer
updates and preserves source checkpoints and the original evaluation artifacts.

The corrected path uses upstream `_get_terrain_column_range` and `is_env_assigned_to_terrain`, assigns each case to the first atlas column of its named terrain, refreshes membership, and asserts membership, pit activity, and assigned origin. The output manifest records the full terrain-to-column mapping and assertion flags.

Guard measurement snapshots `joint_pos_target` during the final physics substep. Terminal case records include `terminal_target` and reset metadata, so post-step reset synchronization cannot overwrite the terminal target evidence.

Strict pit success remains owned by the existing physical/outside/stable gates.
The evaluator uses the shared `update_geometric_climb` helper for an independent
0.2 s geometric milestone (`body_clear`, `all_support`, `upright`, no
physical/outside). Strict parking still requires the original 2 seconds; it is
not replaced by the shorter geometric milestone.

The legacy-named `reset_occurred_after_capture` metadata means the reset callback
and target synchronization ran after the physics capture. `retain_terminal`
suppresses the full scene/manager reset. It does not claim a full physical reset.
The callback count is compared with the count captured at the last physics
substep, so the initial full reset is not counted as a terminal event.

Local checks:

```powershell
python -m pytest -q test_audit_helpers.py
python -m py_compile evaluate_corrected.py
```

Observed checks: 2 shared-helper regression tests passed; Ruff passed for the
corrected evaluator, shared helper, tests, and driver. Runtime behavior remains
subject to the remote paired results; these code checks are not policy acceptance.
