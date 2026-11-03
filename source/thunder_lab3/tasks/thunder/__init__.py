"""Thunder task registration; implementation imports remain owned by the runtime bootstrap."""
import json
import os
import gymnasium as gym

TASK_ID = "Thunder-Newpose-Newton3-v0"
FRESH_TASK_ID = "Thunder-Newpose-Newton3-Fresh-v0"


def make_env(cfg, **kwargs):
    """Apply explicit candidates after configuration composition, before official construction."""
    from isaaclab.envs import ManagerBasedRLEnv
    overrides = json.loads(os.environ.get("NEWPOSE_CANDIDATE_OVERRIDES", "{}"))
    allowed = {"gradual_boxes", "height_weight", "height_options", "hip_extra_weight"}
    if not isinstance(overrides, dict) or not set(overrides) <= allowed:
        raise ValueError("Unsupported Thunder candidate configuration override")
    for name, value in overrides.items():
        setattr(cfg, name, value)
    seed=os.environ.get("NEWPOSE_SEED")
    if seed is not None:
        seed=int(seed)
        if not 0<=seed<2**32:
            raise ValueError("NEWPOSE_SEED must be in [0, 2**32)")
        cfg.seed=seed
    cfg.apply_candidate_features()
    return ManagerBasedRLEnv(cfg=cfg, **kwargs)


gym.register(
    id=TASK_ID,
    entry_point="thunder_lab3.tasks.thunder:make_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "thunder_lab3.tasks.thunder.env_cfg:ThunderNewposeEnvCfg",
        "play_env_cfg_entry_point": "thunder_lab3.tasks.thunder.env_cfg:ThunderNewposePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "thunder_lab3.tasks.thunder.rsl_rl_cfg:ThunderMoeCtsRunnerCfg",
    },
)

gym.register(
    id=FRESH_TASK_ID,
    entry_point="thunder_lab3.tasks.thunder:make_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "thunder_lab3.tasks.thunder.env_cfg:ThunderFreshEnvCfg",
        "play_env_cfg_entry_point": "thunder_lab3.tasks.thunder.env_cfg:ThunderFreshPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "thunder_lab3.tasks.thunder.rsl_rl_cfg:ThunderFreshMoeCtsRunnerCfg",
    },
)
