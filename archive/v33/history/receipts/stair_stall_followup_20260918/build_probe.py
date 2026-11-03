"""Add read-only request/reward capture to the previously validated probe."""
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = (HERE.parent / "stair_failure_peak_20260918/probe_failure.py").read_text()


def replace(before, after):
    global source
    assert source.count(before) == 1, before
    source = source.replace(before, after)


replace('    raw = FirstEpisodeEnv(cfg=cfg)', '''    import guard_actions
    original_bound = guard_actions.bounded_position_target
    capture = {}

    def observe_bound(requested, previous, lower, upper, max_delta):
        capture["requested"] = requested.clone()
        return original_bound(requested, previous, lower, upper, max_delta)

    guard_actions.bounded_position_target = observe_bound
    raw = FirstEpisodeEnv(cfg=cfg)''')
replace('        trace_shapes = {', '''        leg_term = raw.action_manager.get_term("joint_pos")
        leg_joint_names = list(leg_term._joint_names)
        reward_names = list(raw.reward_manager.active_terms)
        trace_shapes = {
            "requested_leg_target_rad": (12,),
            "reward_weighted_rate": (len(reward_names),),''')
replace('                control_values = {', '''                control_values = {
                    "requested_leg_target_rad": capture["requested"],
                    "reward_weighted_rate": raw.reward_manager._step_reward,''')
replace('            joint_names=np.array(names), wheel_body_names=np.array(wheel_body_names),', '''            joint_names=np.array(names), wheel_body_names=np.array(wheel_body_names),
            leg_action_joint_names=np.array(leg_joint_names), reward_names=np.array(reward_names),''')
replace('                      joint_names=names, hard_limits_rad=hard[0].cpu().tolist(),', '''                      leg_action_joint_names=leg_joint_names,
                      reward_weights={name: raw.reward_manager.get_term_cfg(name).weight for name in reward_names},
                      request_capture="After action scaling/base clip, immediately before soft-bound and rate projection; action-term joint order. No mutation.",
                      reward_capture="Weighted instantaneous reward rates before dt; diagnostic configuration weights, not assumed identical to formal reward curricula.",
                      joint_names=names, hard_limits_rad=hard[0].cpu().tolist(),''')
(HERE / "probe_request.py").write_text(source)

