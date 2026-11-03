"""Thunder task registration; implementation imports remain owned by the runtime bootstrap."""
import json
import os
import gymnasium as gym

TASK_ID = "Thunder-Newpose-Newton3-v0"


def make_env(cfg, **kwargs):
    """Apply explicit candidates after configuration composition, before official construction."""
    from isaaclab.envs import ManagerBasedRLEnv
    overrides = json.loads(os.environ.get("NEWPOSE_CANDIDATE_OVERRIDES", "{}"))
    allowed = {"gradual_boxes", "height_weight", "height_options", "hip_extra_weight"}
    if not isinstance(overrides, dict) or not set(overrides) <= allowed:
        raise ValueError("Unsupported Thunder candidate configuration override")
    for name, value in overrides.items():
        setattr(cfg, name, value)
    cfg.apply_candidate_features()
    return ManagerBasedRLEnv(cfg=cfg, **kwargs)


gym.register(
    id=TASK_ID,
    entry_point="thunder_lab3:make_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "thunder_lab3.env_cfg:ThunderNewposeEnvCfg",
        "play_env_cfg_entry_point": "thunder_lab3.env_cfg:ThunderNewposePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "thunder_lab3.rsl_rl_cfg:ThunderMoeCtsRunnerCfg",
    },
)
