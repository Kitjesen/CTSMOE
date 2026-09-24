"""Throwaway live simulator integration probe; never a formal training source."""
import json
from pathlib import Path

from terrain_allocation import reassign
from terrain_allocation_runtime import AdaptiveAllocation
import train

original = AdaptiveAllocation.check
calls = 0


def probe(self, iteration):
    global calls
    calls += 1
    if calls == 1:
        weights = dict(self.course.weights)
        weights['pit'] -= .02
        weights['wave'] += .02
        terrain = self.env.scene.terrain
        levels, columns, receipt = reassign(terrain.terrain_levels.cpu().tolist(),
            terrain.terrain_types.cpu().tolist(), self.names, self.groups, weights, seed=iteration)
        self.pending = {i:(levels[i], columns[i]) for i,_ in receipt['moves']}
        self.course.weights = weights
        return dict(test_only=True, pending_moves=len(self.pending))
    if calls == 16:
        from mixed_task import validate_mixed_runtime
        assert self.moves_applied > 0, 'No reset-boundary migration exercised'
        counts = self.snapshot()['actual_counts']
        checks = validate_mixed_runtime(self.env, counts['pit'])
        assert checks['pit_rewards_zero_on_old']
        result = dict(test_only=True, moves_applied=self.moves_applied,
                      pending_moves=len(self.pending), counts=counts, checks=checks)
        Path(self.env.cfg.log_dir, 'RESET_BOUNDARY_VERIFIED.json').write_text(json.dumps(result, indent=2))
        print('RESET_BOUNDARY_VERIFIED', json.dumps(result), flush=True)
    return original(self, iteration)


AdaptiveAllocation.check = probe
train.main()
