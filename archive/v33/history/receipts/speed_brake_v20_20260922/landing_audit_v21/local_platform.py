"""Local geometric/support diagnostics, independent of the policy reward gate."""
import torch


def platform_from_probes(heights):
    if heights.ndim != 2 or heights.shape[1] != 9:
        raise ValueError('Expected nine authored-platform probe heights per environment')
    if not bool(torch.isfinite(heights).all()):
        raise ValueError('Platform geometry probe missed terrain')
    if not bool(((heights.amax(-1) - heights.amin(-1)) <= .01).all()):
        raise ValueError('Authored outer platform is not flat within one centimeter')
    return heights.median(-1).values


def local_platform_support(local_hull, wheel_ground_z, wheel_clearance, wheel_force_z, platform_z):
    hull_ok = torch.isfinite(local_hull).all(dim=(-1, -2))
    hull_ok &= (local_hull[..., 0].amin(-1) >= 2.9) & (local_hull[..., 0].amax(-1) <= 4.3)
    hull_ok &= local_hull[..., 1].abs().amax(-1) <= 4.3
    height_ok = torch.isfinite(wheel_ground_z) & ((wheel_ground_z - platform_z[:, None]).abs() <= .03)
    contact = height_ok & torch.isfinite(wheel_force_z) & (wheel_force_z > 5.)
    contact &= torch.isfinite(wheel_clearance) & (wheel_clearance.abs() < .04)
    count = contact.sum(-1)
    zmax = torch.where(contact, wheel_ground_z, -torch.inf).amax(-1)
    zmin = torch.where(contact, wheel_ground_z, torch.inf).amin(-1)
    coplanar = (count >= 3) & ((zmax - zmin) <= .05)
    return dict(hull=hull_ok, contact=contact, count=count,
                local_level=coplanar, supported=hull_ok & coplanar)


def update_dwell(previous, supported):
    """Count control steps exactly: 100 x 20ms, avoiding float timer drift."""
    hold = torch.where(supported, previous + 1, torch.zeros_like(previous))
    return hold, hold >= 100


def phase_ids(edge, hull, supported, completed):
    """0 climb, 1 last-step, 2 hull-on-platform, 3 supported, 4 support-dwell."""
    phase = edge.to(torch.int64)
    phase = torch.where(hull, torch.full_like(phase, 2), phase)
    phase = torch.where(supported, torch.full_like(phase, 3), phase)
    return torch.where(supported & completed, torch.full_like(phase, 4), phase)
