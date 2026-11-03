"""Read actual generated stair mesh heights using the evaluation environment setup."""
import json
import os
from pathlib import Path
import sys
import traceback

root = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
audit = root / 'recording/stair_geometry_20260918'
sys.path[:0] = [str(root / 'code'), str(root / 'upstream/source/rsl_rl'),
               str(root / 'upstream/source/robot_lab'), str(root / 'thunder/source/doso_train')]
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaaclab.app import AppLauncher  # noqa: E402 -- launcher requires environment setup first.
app = AppLauncher(headless=True, device='cuda:0').app
try:
    import numpy as np
    import torch
    from config import make_config
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.mdp.terrains import PerSubTerrainSlopeThresholdGenerator

    class MeasuredGenerator(PerSubTerrainSlopeThresholdGenerator):
        def __init__(self, cfg, device):
            self.records = []
            self.seed_used = int(cfg.seed if cfg.seed is not None else np.random.get_state()[1][0])
            super().__init__(cfg, device)

        def _get_terrain_mesh(self, difficulty, cfg):
            self.current_difficulty = float(difficulty)
            return super()._get_terrain_mesh(difficulty, cfg)

        def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
            if col in (6, 7, 8, 11, 12) and row in (3, 6, 9):
                heights = np.unique(np.round(mesh.vertices[:, 2], 6))
                rises = np.diff(heights)
                assert len(rises) and np.allclose(rises, rises[0], atol=1e-5)
                self.records.append(dict(level=row, column=col, difficulty=self.current_difficulty,
                    step_height_m=float(rises[0]), number_of_rises=int(len(rises)),
                    height_difference_m=float(heights[-1]-heights[0]),
                    height_values_m=heights.tolist(), configured_step_width_m=sub_terrain_cfg.step_width,
                    discrete_step_width_m=int(sub_terrain_cfg.step_width/sub_terrain_cfg.horizontal_scale)*sub_terrain_cfg.horizontal_scale))
            return super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)

    torch.set_num_threads(2)
    cfg = make_config(162, root / 'cache/usd')
    cfg.seed = 42
    cfg.log_dir = str(audit)
    cfg.scene.terrain.terrain_generator.class_type = MeasuredGenerator
    # Record metadata from the same scene construction before any rollout.
    original_init = MeasuredGenerator.__init__
    instances = []

    def captured_init(self, cfg, device):
        original_init(self, cfg, device)
        instances.append(self)

    MeasuredGenerator.__init__ = captured_init
    raw = Go2Env(cfg=cfg)
    generator = instances[0]
    result = dict(seed_used=generator.seed_used, records=generator.records,
                  scope='Mesh replay of the same 162-env/seed-42 terrain construction; no policy rollout or training')
    (audit / 'geometry.json').write_text(json.dumps(result, indent=2))
    (audit / 'status.json').write_text(json.dumps({'state': 'complete'}))
    raw.close()
except BaseException as error:
    traceback.print_exc()
    (audit / 'status.json').write_text(json.dumps({'state': 'failed', 'error': str(error)}))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
finally:
    app.close()
