"""Flat-only continuous nominal-pose repair using the upstream L1 functions."""


def flat_mask(env):
    command = env.command_manager.get_term('base_velocity')
    return command.terrain_idxs == command.terrain_type2idx['flat']


def amplify_on_flat(env, cost):
    # Two times the upstream cost on flat, unchanged on every other terrain.
    # This mask has no command-speed, scanner-relief, or posture dead band.
    return cost * (1. + flat_mask(env).to(cost.dtype))


def hip_l1(env, command_name, asset_cfg, stand_still_scale, command_threshold):
    from robot_lab.tasks.go2.mdp.rewards import hip_pos_penalty_l1
    return amplify_on_flat(env, hip_pos_penalty_l1(
        env, command_name, asset_cfg, stand_still_scale, command_threshold))


def leg_l1(env, command_name, asset_cfg, stand_still_scale, velocity_threshold, command_threshold):
    from robot_lab.tasks.go2.mdp.rewards import joint_pos_penalty_l1
    return amplify_on_flat(env, joint_pos_penalty_l1(
        env, command_name, asset_cfg, stand_still_scale, velocity_threshold, command_threshold))


def surface_joint_outside_flat(env, kind):
    from surface_posture import reward
    return reward(env, kind) * ~flat_mask(env)


def configure(cfg):
    # Replace the stronger dead-band actual-pose term on flat; do not stack it
    # with the new continuous cost. Keep all non-flat terms as in V6.
    cfg.rewards.hip_pos_penalty_l1.func = hip_l1
    cfg.rewards.joint_pos_penalty_l1.func = leg_l1
    cfg.rewards.surface_joint_posture.func = surface_joint_outside_flat
