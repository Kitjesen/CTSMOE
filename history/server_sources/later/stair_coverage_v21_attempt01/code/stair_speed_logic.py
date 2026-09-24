"""Tensor-only rules for straight stair speed coverage and terrain eligibility."""
import torch


PROFILE = "straight_speed_v1"
SPEED_EDGES = (0.2, 0.4, 0.6, 0.8, 1.0)


def sample_speeds(count, device):
    bins = torch.randint(4, (count,), device=device)
    edges = torch.tensor(SPEED_EDGES, device=device)
    speeds = edges[bins] + torch.rand(count, device=device) * (edges[bins + 1] - edges[bins])
    return bins, speeds


def terrain_decisions(speed, elapsed_s, maximum_radius, terminated, episode_s, radius_threshold):
    """Keep the 4 m threshold; don't grade time-infeasible slow traversals.

    Reserve two seconds and a 25% tracking margin. Actual physical failures still
    count at any speed. Requested distance is the integral of the held command,
    not a sum of commands sampled in advance every five seconds.
    """
    eligible = speed * max(episode_s - 2.0, 0.0) >= radius_threshold * 1.25
    move_up = eligible & (maximum_radius > radius_threshold) & ~terminated
    move_down = (terminated | (eligible & (maximum_radius < 0.5 * speed * elapsed_s))) & ~move_up
    return move_up, move_down, eligible
