"""Select target samples that still represent observable action updates.

The V14 evaluator records ``robot.data.joint_pos_target`` after ``env.step``.
For a case with a termination term, the target at ``end_step`` has already been
replaced by reset-target synchronization.  The action target that was applied
before that reset is not present in the trace, so the overwritten sample must
be excluded from an offline adjacent-step calculation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def observable_target_mask(
    trace_steps: int,
    *,
    end_step: int,
    termination_terms: Sequence[str] | None,
) -> np.ndarray:
    """Return a mask for target samples that retain action-target semantics.

    Samples after ``end_step`` never belong to the case.  When at least one
    termination term is present, the sample at ``end_step`` is also excluded:
    the V14 evaluator's reset-target wrapper has overwritten it with measured
    joint position before the evaluator reads it.
    """

    if trace_steps < 0:
        raise ValueError("trace_steps must be nonnegative")
    if not 0 <= end_step < trace_steps:
        raise ValueError("end_step must index an available trace sample")

    mask = np.zeros(trace_steps, dtype=np.bool_)
    mask[: end_step + 1] = True
    if termination_terms:
        mask[end_step] = False
    return mask


def observable_target_deltas(
    target: np.ndarray,
    initial_q: np.ndarray,
    *,
    end_step: int,
    termination_terms: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observable adjacent deltas and their original trace step ids.

    The first retained action target is compared with ``initial_q``.  If the
    only recorded sample is a reset-overwritten terminal sample, both returned
    arrays are empty because the true pre-reset action target is unavailable.
    """

    target = np.asarray(target)
    initial_q = np.asarray(initial_q)
    if target.ndim != 2:
        raise ValueError("target must have shape [steps, joints]")
    if initial_q.shape != target.shape[1:]:
        raise ValueError("initial_q must have shape [joints]")

    mask = observable_target_mask(
        target.shape[0], end_step=end_step, termination_terms=termination_terms
    )
    step_ids = np.flatnonzero(mask)
    retained = target[mask]
    if retained.shape[0] == 0:
        return np.empty_like(retained), step_ids
    deltas = np.diff(retained, axis=0, prepend=initial_q[None, :])
    return deltas, step_ids
