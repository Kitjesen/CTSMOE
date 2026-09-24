"""Reproduce the failed zero-update preflight with full infeasible-path evidence."""
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'code'))
import flat_boundary

original = flat_boundary.contain_goals


def capture(position, yaw, goals, lower, upper):
    try:
        return original(position, yaw, goals, lower, upper)
    except RuntimeError:
        import torch
        lo = torch.minimum(lower.expand_as(position), position)
        hi = torch.maximum(upper.expand_as(position), position)
        valid = torch.zeros(len(goals), device=goals.device, dtype=torch.bool)
        for sx, sy in ((1., 1.), (-1., 1.), (1., -1.), (-1., -1.)):
            candidate = goals.clone()
            candidate[:, 0] *= sx
            candidate[:, 1] *= sy
            track = flat_boundary.paths(position, yaw, candidate)
            valid |= ((track >= lo[:, None]-1e-6) & (track <= hi[:, None]+1e-6)).all(dim=(1, 2))
        data = dict(position=position[~valid].cpu().tolist(), yaw=yaw[~valid].cpu().tolist(),
                    goals=goals[~valid].cpu().tolist(), lower=lower.cpu().tolist(),
                    upper=upper.cpu().tolist())
        (HERE/'BOUNDARY_FAILURE_DIAGNOSTIC.json').write_text(json.dumps(data, indent=2))
        print('BOUNDARY_FAILURE_DIAGNOSTIC', json.dumps(data), flush=True)
        raise


flat_boundary.contain_goals = capture
runpy.run_path(str(HERE/'preflight.py'), run_name='__main__')
