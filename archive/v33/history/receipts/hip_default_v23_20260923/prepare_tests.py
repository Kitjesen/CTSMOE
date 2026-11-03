from pathlib import Path

HERE = Path(__file__).resolve().parent
text = (HERE.parent/'hip_guard_v22_20260923/test_hip_guard.py').read_text(encoding='utf-8')
text = text.replace("'candidate_attempt03/code'", "'candidate/code'")
text = text.replace('hip.HipGuardedJointPositionAction(', 'hip.GuardedJointPositionAction(')
start = text.index('def test_inward_requests_are_capped_but_outward_room_is_retained')
end = text.index('def test_subset_reset_does_not_teleport_actual_joints', start)
text = text[:start] + '''def test_default_is_not_a_limit_and_original_bounds_remain(action):
    obj, robot, hard = action
    before_state = robot.data.joint_pos.clone()
    signs = torch.tensor([INWARD_SIGNS[f'{side}_hip_joint'] for side in ('fr','fl','rr','rl')])
    hip_ids = [0, 3, 6, 9]
    request = torch.zeros(8, 12)
    request[:, hip_ids] = signs * ((.25 - .12) / .25)
    obj.process_actions(request)
    torch.testing.assert_close(obj._processed_actions[:, hip_ids] * signs, torch.full((8,4), .25))
    request[:, hip_ids] = signs * 20
    for _ in range(3):
        previous = obj._processed_actions.clone()
        obj.process_actions(request)
        assert (obj._processed_actions-previous).abs().max() <= .240001
    torch.testing.assert_close(obj._processed_actions[:, hip_ids] * signs, torch.full((8,4), .369))
    for _ in range(4):
        obj.process_actions(-request)
    torch.testing.assert_close(obj._processed_actions[:, hip_ids] * signs, torch.full((8,4), -.369))
    torch.testing.assert_close(robot.data.joint_pos, before_state, rtol=0, atol=0)
    torch.testing.assert_close(robot.data.soft_joint_pos_limits, hard*.9, rtol=0, atol=0)


def test_configuration_changes_only_four_default_hip_angles(action):
    from hip_guard import configure, GuardedJointPositionAction
    original = {name: INWARD_SIGNS[name] * .20 for name in INWARD_SIGNS}
    original.update(fr_thigh_joint=-.93, fl_thigh_joint=.93, fr_calf_joint=1.96, fl_calf_joint=-1.96)
    initial = SimpleNamespace(joint_pos=original.copy(), pos=(0.,0.,.46))
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(init_state=initial)),
                          actions=SimpleNamespace(joint_pos=SimpleNamespace(class_type=GuardedJointPositionAction)))
    configure(cfg)
    assert initial.joint_pos == original
    assert cfg.scene.robot.init_state.pos == initial.pos
    assert cfg.actions.joint_pos.class_type is GuardedJointPositionAction
    for name,value in cfg.scene.robot.init_state.joint_pos.items():
        assert value == (INWARD_SIGNS[name]*.12 if name in INWARD_SIGNS else original[name])


''' + text[end:]
(HERE/'test_hip_default.py').write_text(text, encoding='utf-8')
