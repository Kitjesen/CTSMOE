"""Instrument actual reward-manager contributions without altering commands."""
from pathlib import Path

p=Path(__file__).parent
s=(p/'transition_probe.py').read_text()
config="""    from startup_repair import configure as configure_startup
    command_class_before = cfg.commands.base_velocity.class_type
    joint_reset_before = cfg.events.reset_robot_joints.func
    configure_startup(cfg, latest['startup_repair_profile'])
    # Reward audit only: preserve the independent start/stop evaluation protocol.
    cfg.commands.base_velocity.class_type = command_class_before
    cfg.events.reset_robot_joints.func = joint_reset_before
    cfg.terminations.startup_window = None
"""
instrument="""    from surface_posture import measure as measure_surface
    original_reward_compute = raw.reward_manager.compute
    reward_capture = {}
    def capture_reward(dt):
        result = original_reward_compute(dt)
        state = measure_surface(raw)
        reward_capture['weighted_reward'] = raw.reward_manager._step_reward.clone()*dt
        reward_capture['reward_total'] = result.clone()
        reward_capture['posture_gates'] = torch.stack((state['weight'],state['height_weight']),dim=-1)
        reward_capture['reward_height'] = state['height_m'].clone()
        reward_capture['reward_command'] = command.commands.clone()
        torch.testing.assert_close(reward_capture['weighted_reward'].sum(-1),result)
        return result
    raw.reward_manager.compute = capture_reward
"""
changes={
    "sys.path[:0]=[receipt['code'],":"sys.path[:0]=[str(a.root/'code_startup_repair_20260920_v4'),",
    "    for name in vars(cfg.curriculum):":config+"    for name in vars(cfg.curriculum):",
    "    obs=env.get_observations()":instrument+"    obs=env.get_observations()",
    "'pre_joint_pos')}":"'pre_joint_pos','weighted_reward','reward_total','posture_gates','reward_height','reward_command')}",
    "            values.update(extra)":"            values.update(extra)\n            values.update(reward_capture)",
    "                  history=metadata,":"                  history=metadata,\n                  reward_names=raw.reward_manager.active_terms,\n                  reward_weights={name:raw.reward_manager.get_term_cfg(name).weight for name in raw.reward_manager.active_terms},",
    "protocol='Transition probe:":"protocol='Actual v4 reward-manager contribution audit on v3/v4 policies; no optimizer updates. Transition probe:",
}
for old,new in changes.items():
    assert s.count(old)==1,old
    s=s.replace(old,new)
(p/'reward_probe.py').write_text(s,encoding='utf-8')
