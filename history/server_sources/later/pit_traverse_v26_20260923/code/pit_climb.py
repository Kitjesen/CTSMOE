"""Climb-first milestone and smooth post-climb stability shaping."""
import math
import torch

PROFILE = "climb_first_v2"


def confirm_climb(hold, seen, supported, clear, upright, invalid, started, dt):
    valid = supported.all(-1) & clear & upright & ~invalid & started
    hold = torch.where(valid, hold + 1, torch.zeros_like(hold))
    confirmed = hold >= math.ceil(.2 / dt)
    event = confirmed & ~seen
    return hold, seen | confirmed, event


def stability_reward(supported, gravity_z, angular_velocity, enabled):
    support = supported.float().mean(-1)
    upright = (-gravity_z).clamp(0, 1).square()
    angular = torch.exp(-angular_velocity.square().sum(-1) / .3**2)
    return support * upright * angular * enabled
