"""Create an isolated, auditable derivative of the existing validated evaluator."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parents[1]/'speed_brake_v20_20260922/landing_audit_v21/evaluate_local_platform.py'
text = source.read_text(encoding='utf-8')


def replace(old, new):
    global text
    if text.count(old) != 1:
        raise ValueError(f'Expected one exact source block: {old[:80]}')
    text = text.replace(old, new)


replace('from focus_protocol import build_focus_cases, brake_factor',
        'from focus_protocol import brake_factor\nfrom release_cases_v2 import build_release_cases as build_focus_cases, load_student_selection as load_selection')
replace('load_selection, source_contracts)', 'source_contracts)')
replace('    return parser.parse_args()', '''    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--perturbed', action='store_true')
    return parser.parse_args()''')
replace('        state[:, 3] = 1.\n        robot.write_root_state_to_sim(state, env_ids=other_ids)', '''        state[:, 3] = 1.
        if args.perturbed:
            sign = torch.where(other_ids % 2 == 0, 1., -1.)
            yaw = sign * (5. * 3.141592653589793 / 180.)
            state[:, 1] += sign * .10
            state[:, 3] = torch.cos(yaw/2.)
            state[:, 6] = torch.sin(yaw/2.)
        robot.write_root_state_to_sim(state, env_ids=other_ids)''')
replace('        group.enable_corruption = False', '        group.enable_corruption = args.perturbed')
replace('    def act(model, observations, teacher):\n        if not teacher:\n            return model.act_inference(observations)', '''    import onnxruntime as ort
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = 2
    ort_options.inter_op_num_threads = 1
    onnx_session = ort.InferenceSession(str(args.onnx), sess_options=ort_options,
                                       providers=['CPUExecutionProvider'])
    parity_peaks = []
    real_observations = []

    def act(model, observations, teacher):
        if not teacher:
            expected = model.act_inference(observations)
            feed = {key: observations[key].cpu().numpy() for key in ('policy', 'single_obs')}
            actual = onnx_session.run(None, {'history': feed['policy'], 'single_obs': feed['single_obs']})[0]
            np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=1e-4, rtol=1e-4)
            parity_peaks.append(float(np.max(np.abs(actual-expected.cpu().numpy()))))
            if len(parity_peaks) % 25 == 1:
                real_observations.append((feed['policy'].copy(), feed['single_obs'].copy(), actual.copy()))
            return torch.from_numpy(actual).to(raw.device)''')
replace('    trace_names_extra = (', '''    profile_x = torch.linspace(0., 4.25, 851, device=raw.device)
    profile_start = origins[:, None, :].repeat(1, len(profile_x), 1)
    profile_start[..., 0] += profile_x
    profile_start[..., 2] += 10.
    profile_direction = torch.zeros_like(profile_start)
    profile_direction[..., 2] = -1.
    profile_z = raycast_mesh(profile_start, profile_direction, mesh)[0][..., 2] - origins[:, 2:3]
    if not bool(torch.isfinite(profile_z).all()):
        raise RuntimeError('Stair profile raycast incomplete')
    np.savez_compressed(args.output/'stair_profiles.npz', x=profile_x.cpu().numpy(),
                        z=profile_z.cpu().numpy())
    position = raw.action_manager.get_term('joint_pos')
    velocity = raw.action_manager.get_term('joint_vel')
    from config import ALL_JOINTS, LEGS, WHEELS
    runtime_contract = dict(policy_leg_order=LEGS, policy_wheel_order=WHEELS, policy_all_order=ALL_JOINTS,
        physics_order=robot.joint_names, dt=raw.step_dt, physics_dt=raw.physics_dt,
        default_q=robot.data.default_joint_pos[0].cpu().tolist(),
        soft_joint_pos_limits=robot.data.soft_joint_pos_limits[0].cpu().tolist(),
        hard_joint_pos_limits=guard.hard[0].cpu().tolist(),
        position_joint_indices=position.joint_indices,
        position_scale=position._scale.cpu().tolist() if torch.is_tensor(position._scale) else position._scale,
        velocity_scale=velocity._scale.cpu().tolist() if torch.is_tensor(velocity._scale) else velocity._scale,
        observation_terms=raw.observation_manager.active_terms,
        observation_term_dimensions={name: [tuple(int(x) for x in dim) for dim in dims]
                                     for name, dims in raw.observation_manager.group_obs_term_dim.items()},
        clip_actions=MoECTSRunnerCfg().clip_actions,
        actor_normalizer=repr(policies[0].actor_obs_normalizer),
        single_normalizer=repr(policies[0].single_obs_normalizer),
        actuator_limits=guard.limits,
        actuator_gains={name: dict(stiffness=a.stiffness, damping=a.damping)
                        for name, a in cfg.scene.robot.actuators.items()})
    (args.output/'RUNTIME_CONTRACT.json').write_text(json.dumps(runtime_contract, indent=2))
    trace_names_extra = (''')
replace('    geom_hold = torch.zeros(', '''    quiet_hold_steps = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.int64)
    traces['quiet_hold_steps'] = []
    geom_hold = torch.zeros(''')
replace('            platform_hold = platform_hold_steps * raw.step_dt', '''            quiet_supported = supported_platform & pit_quiet & pit_upright & landing_stop
            quiet_hold_steps, quiet_done = update_dwell(quiet_hold_steps, quiet_supported)
            # Support alone cannot end a stop trial. Keep all original physical terminations.
            dwell_done = quiet_done & stop_on_landing
            platform_hold = platform_hold_steps * raw.step_dt''')
replace('            values = dict(\n', "            values = dict(\n                quiet_hold_steps=quiet_hold_steps,\n")
replace("                    end_reason = 'supported_platform_dwell'", "                    end_reason = 'quiet_supported_platform_dwell'")
replace('                    end_reason=end_reason, platform_hold_s=float(platform_hold[case_id]),', '''                    end_reason=end_reason, platform_hold_s=float(platform_hold[case_id]),
                    quiet_hold_s=float(quiet_hold_steps[case_id]*raw.step_dt),''')
replace('    base_id = robot.body_names.index("base_link")', '''    np.savez_compressed(args.output/'REAL_OBSERVATIONS.npz',
                        history=np.concatenate([r[0] for r in real_observations]),
                        single_obs=np.concatenate([r[1] for r in real_observations]),
                        actions=np.concatenate([r[2] for r in real_observations]))
    (args.output/'ONNX_PARITY.json').write_text(json.dumps(dict(
        steps=len(parity_peaks), observations=len(parity_peaks)*len(cases),
        max_absolute_error=max(parity_peaks), atol=1e-4, rtol=1e-4,
        actual_simulation_actions='ONNX Runtime CPU', perturbed=args.perturbed), indent=2))
    base_id = robot.body_names.index("base_link")''')
replace('evaluation_revision="local_platform_audit_v21_20260922",', '''evaluation_revision="continuous_stairs_onnx_release_matrix_20260922",
        perturbed=args.perturbed, fixed_case_matrix=True, randomized_success_rate=False,
        perturbation=dict(observation_noise=args.perturbed, initial_yaw_degrees=5. if args.perturbed else 0.,
                          initial_lateral_m=.1 if args.perturbed else 0., pushes=False,
                          mass_friction_randomization=False),''')
replace('"default-pose reset with synchronized targets, no noise or pushes, fixed 10 ms delay. "',
        '"default-joint-pose reset with synchronized targets, fixed 10 ms delay. See explicit perturbation metadata. "')
(HERE/'evaluate_stairs_onnx_v3.py').write_text(text, encoding='utf-8')
