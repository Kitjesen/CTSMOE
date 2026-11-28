"""CTS+MoE runner configuration for Lope locomotion."""
from isaaclab.utils import configclass
from rsl_rl.moe_cts.config import (
    MoeCtsSymmetryCfg,
    RslRlMoeCtsActorCriticCfg,
    RslRlMoeCtsAlgorithmCfg,
    RslRlMoeCtsRunnerCfg,
)
from rsl_rl.utils.robogauge_evaluator import RoboGaugeEvaluatorCfg

from lope_rl_lab.tasks.locomotion.config import (
    CRITIC_DIM,
    CUMULATIVE_UPDATES,
    EXPERIMENT_NAME,
    EXPERTS,
    LATENT_DIM,
    ROLLOUT_STEPS,
)


@configclass
class LopeLocomotionMoeCtsRunnerCfg(RslRlMoeCtsRunnerCfg):
    class_name = "lope_rl_lab.runners.moe_cts_runner:LopeMoeCtsRunner"
    num_steps_per_env = ROLLOUT_STEPS
    max_iterations = CUMULATIVE_UPDATES
    clip_actions = None
    clip_rewards = 10.0
    save_interval = 500
    log_interval = 1
    experiment_name = EXPERIMENT_NAME
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    robogauge = RoboGaugeEvaluatorCfg(enabled=False)
    lope_compat = {
        "cumulative_target": CUMULATIVE_UPDATES,
        "critic_dim": CRITIC_DIM,
        "history_layout": "frame_major_current_last",
        "robot": "lope_rs02_12dof",
    }
    policy = RslRlMoeCtsActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        teacher_encoder_hidden_dims=[512, 256],
        student_encoder_hidden_dims=[512, 256, 256],
        expert_num=EXPERTS,
        latent_dim=LATENT_DIM,
        norm_type="l2norm",
        activation="elu",
    )
    algorithm = RslRlMoeCtsAlgorithmCfg(
        value_loss_coef=1.0,
        load_balance_coef=0.01,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        student_encoder_learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_env_ratio=0.75,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=MoeCtsSymmetryCfg(
            symmetry_class="lope_rl_lab.tasks.locomotion.mdp.symmetry:LopeSymmetry"
        ),
    )
