# V21 two-column stair terminal coverage

This is a geometry-only patch. It does not modify rewards, commands, actuator
limits, terminations, resets, allocations, curriculum grading or model state.
The frozen V19/V20 code is not edited. Importing the module for its NumPy helpers
does not import IsaacLab, PyTorch or `mixed_task`.

## Integration

In the NEW isolated training snapshot, call after `configure_mixed(...)` and
before environment construction:

```python
from stair_transition_coverage import configure as configure_stair_coverage
configure_stair_coverage(cfg)
```

Calling before `configure_mixed` is rejected: `config.make_config` still returns
the original 10-row/20-column layout, and `configure_mixed` would later overwrite
the installed generator class. This patch accepts only the final 22-row/40-column
mixed atlas and the exact original `MixedTerrainGenerator` class. Double
installation or a pre-existing global `phase_landing_kind` is rejected.

After restoring the checkpoint's terrain assignments and installing teacher /
student IDs, record:

```python
from stair_transition_coverage import snapshot
receipt = snapshot(raw)
```

The snapshot is JSON-ready. It contains the immutable profile configuration,
actual coverage counts by learner/type/level, and all 880 tile geometry receipts.
No assignment is changed. Call again when reporting later coverage; per-level
counts then reflect the actual live assignments. Geometry receipts remain fixed.

## Allowed change and validation

The runtime class subclasses `MixedTerrainGenerator` and overrides only the
placement hook `_add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)`.
It applies the frozen `phase_landing` height-cap algorithm before atlas placement
to **stairs_up column 8** and **stairs_down column 11**, across all 22 rows.
This changes 44 tiles. Original terrains actually use levels 0–9; rows 10–21
retain the generator's repeated-level-9 behavior and receive the same cap.

Validation requires 9 m tiles, a 0.5 m sub-terrain border, the pinned column
mapping, and original stair parameters: nominal riser range 0.05–0.257 m,
tread width 0.31 m, center platform width 3 m. All nine terrain identities and
the column order remain intact; zero-proportion unused definitions are allowed.
Each placement must receive the configuration object matching its column.
The completed generator must have exactly 880 receipts and exactly the 44
allowed cells selected.

The actual cap height is measured from the most common rounded surface vertex
height in infinity-norm radius 2.7–2.9 m, matching the original cap helper.
Upstairs uses `min(z, cap)` and downstairs uses `max(z, cap)`. XY and spawn origins
are unchanged. Only degenerate faces of area <= 1e-10 m² and unreferenced
vertices are removed. A selected tile with no outer geometry to cap is rejected.

Every tile records canonical numeric SHA256 before/after geometry, measured
before atlas placement. Nonselected tiles must have identical hashes. Selected
tiles additionally record exact before/after hashes of inner vertex positions,
and of actual output triangles wholly inside radius < 2.7 m, with nondegenerate
inner faces preserved. Compacting the vertex index does not invalidate that
triangle-coordinate proof. Face/vertex counts and removed degenerate faces are
recorded. The source mesh is copied before modification.

Each selected tile also records measured cap and center heights and four
cardinal surface profiles before/after. Profiles use the nearest sampled
centerline and its upper surface envelope, compressed into constant-height
runs; their lateral offset is stated. Measured inner riser heights are listed.
These actual mesh measurements, rather than nominal configuration alone, are
the geometry receipt. They are not collision-contact or policy-performance
proof. Hashes of nonselected meshes can additionally be compared with a source
generation using the same pinned code, configuration and random seed.

The added landing preserves internal risers and tread geometry but **removes
outer steps and reduces total flight rise**. It must not be described as an
entirely unchanged stair challenge. Original stair columns 6,7,9,10 and 12 remain
unchanged. Coverage is determined by restored column assignments, not an
invented nominal environment percentage. The count helper rejects original
terrain assignments above level 9.

## Deliberate limits

The original root-progress > 4 m route termination and stair speed sampling
remain unchanged. Fast traversal may expose less than the full posture recovery
period on the platform. This patch adds moving terminal-transition experience;
it does not implement platform-stop commands or 2-second supported dwell.
Existing height/posture courses advance normally with the restored clock.
No `phase_transition.configure()` call or old phase-profile migration is made.

Root integration owns the isolated train entry, exact source33865 optimizer and
curriculum restoration, zero-update simulator preflight, the 958-update stage,
budget accounting, evaluation and deployment. Local helper tests cannot validate
the actual remote IsaacLab class import or collision mesh.

## Local tests

```text
python -m pytest -q -p no:cacheprovider coverage_patch
ruff check coverage_patch
```

Tests use real NumPy triangular meshes and a small generator double; no GPU,
SSH or IsaacLab launch occurs. They cover both stair directions, inherited cap
semantics, inner geometry preservation, actual platform measurements, invalid
configuration, duplicate/global-capping rejection, all 880 placement decisions,
unchanged origins and other configuration objects, and per-learner coverage.
