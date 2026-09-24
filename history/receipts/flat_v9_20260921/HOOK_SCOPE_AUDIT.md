# Stop hook scope audit

The reported files are outside this task's CTS/MoE changes. They were read but not modified. No hook rules or exclusions were changed.

- `brain/train/flat/reports/20260919-motion-refine/evidence/exit-diagnosis/simulation_app.py:831`: NVIDIA-labelled SimulationApp shutdown evidence copy. The caught RuntimeError is around optional Tracy profiler acquisition/shutdown, followed by the normal app shutdown. This is not a robot control or reward safety bypass added by this task. The available header establishes attribution, not upstream byte parity; do not rewrite historical evidence to satisfy a keyword check.
- `modules/doso_motor/v0.7/DS04_V07_A07_齿轮运动与啮合核查/deps/dateutil/zoneinfo/__init__.py:72`: existing comment explains compatibility with the public module-level timezone API while sharing a stateful ZoneInfo instance. Package version and installed RECORD hash verification are in hook_scope_audit.json.
- `modules/doso_motor/v0.7/DS04_V07_A07_齿轮运动与啮合核查/deps/mpl_toolkits/axisartist/grid_finder.py:164`: existing documentation and explicit isinstance branch support both Matplotlib Formatter.format_ticks and the legacy axisartist formatter call signature. Package RECORD hash verification is in hook_scope_audit.json.

The actionable CTS/MoE change remains flat_continuous.py plus the reviewed V9 profile integration in change.diff. Those files have passed the recorded relevant unit tests and full-environment integration checks. This audit records scope and compatibility rationale; it does not claim the broad repository stop hook has been rerun or cleared.

## Follow-up: explicit adjacent compatibility annotations

The repeated hook explicitly requested rationale near the code. Three comment-only annotations were added at the reported locations. Original bytes are preserved in hook_original_sources.zip; hook_annotation_verified.json confirms identical Python ASTs before and after, including docstrings. The dependency RECORD hashes describe the original bytes, not the now-annotated files. No runtime behavior, training configuration, or hook rule was changed. The earlier read-only disposition above describes the initial audit, superseded by this annotation step.
