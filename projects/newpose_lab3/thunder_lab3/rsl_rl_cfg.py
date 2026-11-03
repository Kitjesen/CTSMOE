"""Thunder CTS+MoE configuration using the pinned Go2 lab3.0 vendored runner interfaces."""
from isaaclab.utils import configclass
from rsl_rl.moe_cts.config import (
    RslRlMoeCtsActorCriticCfg,
    RslRlMoeCtsAlgorithmCfg,
    RslRlMoeCtsRunnerCfg,
)
from rsl_rl.utils.robogauge_evaluator import RoboGaugeEvaluatorCfg


@configclass
class ThunderMoeCtsRunnerCfg(RslRlMoeCtsRunnerCfg):
    class_name = "compat.runner:ThunderMoeCtsRunner"
    num_steps_per_env = 24
    # Upstream learn() treats this as additional updates. The wrapper requires an explicit budget.
    max_iterations = 0
    clip_actions = None
    clip_rewards = None
    save_interval = 100
    log_interval = 1
    experiment_name = "NEWPOSE_NEWTON3_CANDIDATE"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    robogauge = RoboGaugeEvaluatorCfg(enabled=False)
    newpose_compat = {
        "cumulative_target": 20000,
        "critic_dim": 309,
        "history_layout": "frame_major_current_last",
    }
    policy = RslRlMoeCtsActorCriticCfg(
        init_noise_std=1.,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        teacher_encoder_hidden_dims=[512, 256],
        student_encoder_hidden_dims=[512, 256, 256],
        expert_num=8,
        latent_dim=32,
        norm_type="l2norm",
        activation="elu",
    )
    algorithm = RslRlMoeCtsAlgorithmCfg(
        value_loss_coef=1.,
        load_balance_coef=.01,
        use_clipped_value_loss=True,
        clip_param=.2,
        entropy_coef=.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.e-3,
        student_encoder_learning_rate=1.e-3,
        schedule="adaptive",
        gamma=.99,
        lam=.95,
        betas=(.9, .999),
        weight_decay=0.,
        desired_kl=.01,
        max_grad_norm=1.,
        teacher_env_ratio=.75,
        normalize_advantage_per_mini_batch=False,
        # Go2's default mirror mapping has 12 actions and must never reach this 16-action robot.
        symmetry_cfg=None,
    )
