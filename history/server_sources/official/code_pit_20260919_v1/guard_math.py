"""Tensor-only target bounds and normalized actuator costs.

The last dimension is joints; all preceding dimensions are independent samples.
Callers must supply floating-point state tensors, positive motor limits, ordered
position bounds, and a nonnegative target increment. No physical state is changed.
"""

import torch


def bounded_position_target(requested, previous, lower, upper, max_delta):
    """Project targets into physical bounds and a per-update change interval.

    Bounds and ``max_delta`` may be scalars or tensors broadcastable to the
    targets. An out-of-range previous target is first projected into the physical
    interval; the change limit is measured from that projected target. This also
    permits recovery after a reset or a change to the configured joint bounds.
    Inputs, including the caller's previous-target buffer, are left untouched.
    """
    lower = torch.as_tensor(lower, dtype=requested.dtype, device=requested.device)
    upper = torch.as_tensor(upper, dtype=requested.dtype, device=requested.device)
    max_delta = torch.as_tensor(max_delta, dtype=requested.dtype, device=requested.device)
    previous = torch.clamp(previous, min=lower, max=upper)
    requested = torch.clamp(requested, min=lower, max=upper)
    delta = torch.clamp(requested - previous, min=-max_delta, max=max_delta)
    return torch.clamp(previous + delta, min=lower, max=upper)


def guard_costs(applied_torque, computed_torque, velocity, torque_limit, velocity_limit, load_reference=None):
    """Return per-sample costs using separate peak and load references.

    Speed cost starts above 80% of the configured limit. Saturation uses the
    signed computed-minus-applied torque, so opposite-sign discrepancies remain
    visible. Load reference defaults to the peak limit for legacy profiles.
    Costs have no upper clipping: larger excursions keep their penalty.
    """
    load_reference = torque_limit if load_reference is None else load_reference
    return {
        "torque_mean_square": (applied_torque / load_reference).square().mean(dim=-1),
        "speed_excess": torch.relu(velocity.abs() / velocity_limit - 0.8).square().sum(dim=-1),
        "saturation": ((computed_torque - applied_torque) / torque_limit).square().mean(dim=-1),
    }
