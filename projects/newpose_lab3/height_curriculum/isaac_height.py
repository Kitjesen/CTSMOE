"""Isaac manager adapter, imported only when the height candidate is enabled."""
import torch
from isaaclab.managers import ManagerTermBase

from .height import HeightReference, torch_view


class CroppedGroundHeight(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        sensor = env.scene.sensors[p["sensor_cfg"].name]
        if getattr(sensor.cfg, "ray_alignment", None) != "yaw":
            raise ValueError("height candidate requires a yaw-aligned scanner")
        if sensor.cfg.update_period > env.step_dt + 1e-9:
            raise ValueError("height scanner must update at least once per policy step")
        # Crop the yaw-aligned ray pattern, not the variable hit positions.
        local = torch_view(sensor.ray_starts)[0, :, :2]
        tolerance = max(1e-6, float(getattr(sensor.cfg.pattern_cfg, "resolution", 0)) * 1e-5)
        x, y = p["x_range"], p["y_range"]
        self.crop = ((local[:, 0] >= x[0] - tolerance) & (local[:, 0] <= x[1] + tolerance)
                     & (local[:, 1] >= y[0] - tolerance) & (local[:, 1] <= y[1] + tolerance))
        if not bool(self.crop.any()):
            raise ValueError("height reward crop contains no scanner rays")
        base = torch_view(env.scene[p["asset_cfg"].name].data.root_pos_w)
        self.reference = HeightReference(env.num_envs, base.device, base.dtype,
                                         env.step_dt, p["options"])

    def reset(self, env_ids=None):
        self.reference.reset(env_ids)

    def __call__(self, env, options, asset_cfg, sensor_cfg, x_range, y_range):
        base_z = torch_view(env.scene[asset_cfg.name].data.root_pos_w)[:, 2]
        hits_z = torch_view(env.scene.sensors[sensor_cfg.name].data.ray_hits_w)[:, self.crop, 2]
        return self.reference.step(base_z, hits_z)
