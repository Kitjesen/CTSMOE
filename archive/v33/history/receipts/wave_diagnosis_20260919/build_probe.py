"""Derive a wave-only, single-factor diagnostic from the validated motion probe."""

from pathlib import Path


folder = Path(__file__).resolve().parent
source = (folder.parent / 'motion_quality_20260918/probe_motion.py').read_text()


def replace(before, after):
    global source
    assert source.count(before) == 1, before
    source = source.replace(before, after)


replace('for name in ("wave", "stairs_up", "stairs_down"):', 'for name in ("wave",):')
replace('for level in (3, 6, 9):', 'for level in (0, 3, 6, 9):')
replace('        assert count == 162 and len(stair_trace_ids) == 81 and len(names) == 16',
        '        assert count == 72 and len(stair_trace_ids) == 72 and len(names) == 16')
replace('        stair_trace_ids = [i for i, c in enumerate(cases) if c["learner"] == "student"]',
        '        stair_trace_ids = list(range(count))')
replace('    columns = saved["terrain_column_names"]',
        '    from policy_contract import checkpoint_profile\n'
        '    assert checkpoint_profile(saved) == args.actuator_profile\n'
        '    columns = saved["terrain_column_names"]')
replace('    import guard_actions',
        '    cfg.events.reset_base.params["pose_range"]["z"] = (args.reset_z, args.reset_z)\n'
        '    import guard_actions')
replace('        def command_update():\n            original_command_update()',
        '        def command_update():\n            original_command_update()\n'
        '            if args.settle_seconds:\n'
        '                command.commands[:, 0] = torch.where(\n'
        '                    raw.episode_length_buf * raw.step_dt < args.settle_seconds, 0.0, speeds)')
replace('        leg_joint_names = list(leg_term._joint_names)',
        '        leg_joint_names = list(leg_term._joint_names)\n'
        '        if args.target_rate is not None:\n'
        '            leg_term._max_delta = args.target_rate * raw.step_dt\n'
        '        # Read-only raycasts under wheel centers; gap uses nominal 0.093 m radius.\n'
        '        from isaaclab.utils.warp import raycast_mesh\n'
        '        initial_points = torch.cat((robot.data.root_pos_w[:, None], robot.data.body_link_pos_w[:, wheel_body_ids]), 1).clone()\n'
        '        ray_starts = initial_points.clone()\n'
        '        ray_starts[..., 2] += 10.0\n'
        '        ray_directions = torch.zeros_like(ray_starts)\n'
        '        ray_directions[..., 2] = -1.0\n'
        '        scanner = raw.scene["height_scanner"]\n'
        '        hits = raycast_mesh(ray_starts, ray_directions, scanner.meshes[scanner.cfg.mesh_prim_paths[0]])[0]\n'
        '        assert torch.isfinite(hits).all()\n'
        '        start_geometry = dict(points_w=initial_points.cpu().tolist(), terrain_hits_w=hits.cpu().tolist(),\n'
        '                              wheel_bottom_vertical_gap_m=(initial_points[:, 1:, 2] - hits[:, 1:, 2] - 0.093).cpu().tolist(),\n'
        '                              root_height_above_local_surface_m=(initial_points[:, 0, 2] - hits[:, 0, 2]).cpu().tolist())\n'
        '        early = []\n'
        '        base_sensor_id = contact.body_names.index("base_link")')
replace('        def measure(dt):\n            update_scene(dt)',
        '        def measure(dt):\n            update_scene(dt)\n'
        '            if len(early) < 200:\n'
        '                early.append(torch.cat((robot.data.root_pos_w - initial[:, :3],\n'
        '                    robot.data.root_lin_vel_b, robot.data.joint_vel, robot.data.applied_torque,\n'
        '                    torch.linalg.vector_norm(contact.data.net_forces_w[:, [base_sensor_id] + wheel_sensor_ids], dim=-1),\n'
        '                    active[:, None]), dim=-1).clone())')
replace('        import numpy as np\n        np.savez_compressed(args.output / "selected_traces.npz",',
        '        import numpy as np\n'
        '        (args.output / "initial_geometry.json").write_text(json.dumps(start_geometry, indent=2))\n'
        '        np.savez_compressed(args.output / "early_200hz.npz", values=torch.stack(early).cpu().numpy(),\n'
        '                            fields=np.array(["dx", "dy", "dz", "vx", "vy", "vz"] +\n'
        '                            ["speed_"+n for n in names] + ["torque_"+n for n in names] +\n'
        '                            ["contact_base"] + ["contact_"+n for n in wheel_body_names] + ["active"]))\n'
        '        np.savez_compressed(args.output / "selected_traces.npz",')
replace('        result = dict(actuator_profile=args.actuator_profile,',
        '        result = dict(ablation=dict(target_rate=args.target_rate, reset_z=args.reset_z, settle_seconds=args.settle_seconds),\n'
        '                      actuator_profile=args.actuator_profile,')
replace('    parser.add_argument("--mode", choices=("baseline", "heading"), default="baseline")',
        '    parser.add_argument("--target-rate", type=float, default=None)\n'
        '    parser.add_argument("--reset-z", type=float, default=0.0)\n'
        '    parser.add_argument("--settle-seconds", type=float, default=0.0)\n'
        '    parser.add_argument("--mode", choices=("baseline", "heading"), default="baseline")')
source = source.replace('over the unchanged 162-case probe.', 'over a wave-only 72-case grid; explicit isolated ablations.')
source = source.replace('"unchanged_physics": "No root/joint state, action, reward, actuator limit, sensor configuration, randomization or case setup is changed by added observations.",',
                        '"diagnostic_ablation": dict(target_rate=args.target_rate, reset_z=args.reset_z, settle_seconds=args.settle_seconds),')
(folder / 'probe_wave.py').write_text(source)
