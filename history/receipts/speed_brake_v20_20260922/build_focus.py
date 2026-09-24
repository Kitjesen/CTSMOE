"""Add diagnostic suites while preserving the frozen 332+80 evaluation driver."""
from pathlib import Path
from build_runtime import replace

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'omni_repair_v19_20260922'


def main():
    s = (OLD/'evaluate_measured.py').read_text(encoding='utf-8')
    s = replace(s, 'from early_protocol import (build_cases,', 'from early_protocol import (build_cases as original_build_cases,')
    s = replace(s, 'from audit_helpers import assign_columns, update_geometric_climb', 'from audit_helpers import assign_columns, update_geometric_climb\nfrom focus_protocol import build_focus_cases, brake_factor')
    s = replace(s, 'choices=("terrain83", "flat25")', 'choices=("brake25", "stairend")')
    s = replace(s, '    cases = build_cases(selection, args.suite)', '    cases = build_focus_cases(selection, args.suite, original_build_cases)')
    s = s.replace('args.suite == "flat25"', 'args.suite == "brake25"')
    s = replace(s, '    cfg.rewards.lin_vel_z_l2.weight = 0.', '''    cfg.rewards.lin_vel_z_l2.weight = 0.
    if args.suite == 'stairend':
        cfg.terminations.stair_route_exit = None
        cfg.scene.terrain.terrain_generator.use_cache = False
        for name in ('stairs_up', 'stairs_down'):
            cfg.scene.terrain.terrain_generator.sub_terrains[name].phase_landing_kind = name''')
    s = replace(s, '    def update_commands():', '''    landing_stop = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)
    stop_on_landing = torch.tensor([c.get('landing_mode') == 'stop' for c in cases], device=raw.device)

    def update_commands():''')
    s = replace(s, '        command.commands[:] = velocities * (elapsed >= settle).float()[:, None]', '''        factor = brake_factor(elapsed) if args.suite == 'brake25' else torch.ones_like(elapsed)
        command.commands[:] = velocities * ((elapsed >= settle).float()*factor)[:, None]
        command.commands[landing_stop] = 0.''')
    s = replace(s, '"phase_weight", "phase_ready", "phase_stable_s")', '"phase_weight", "phase_ready", "phase_stable_s", "world_velocity",\n                   "stair_edge_window", "stair_platform_hull", "stair_platform_supported")')
    s = replace(s, '    geom_hold = torch.zeros', '    platform_hold = torch.zeros(raw.num_envs, device=raw.device)\n    platform_seen = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)\n    geom_hold = torch.zeros')
    s = replace(s, '            zeros = torch.zeros(raw.num_envs, device=raw.device)', '''            from phase_transition import measure as measure_phase
            phase = measure_phase(raw)
            hull = pit.geometry.world_points(data.joint_pos, data.root_state_w[:, :7])
            local = hull - origins[:, None, :]
            # Straight cases all move +X. Validate the full collision hull and
            # wheel support on the actual outer landing, not just root distance.
            on_platform = (local[..., 0].amin(1) >= 2.9) & (local[..., 0].amax(1) <= 4.3)
            on_platform &= local[..., 1].abs().amax(1) <= 4.3
            supported_platform = on_platform & phase['ready'] & phase['stairs']
            edge_window = (local[..., 0].amax(1) >= 2.6) & phase['stairs']
            platform_seen |= supported_platform
            platform_hold = torch.where(supported_platform, platform_hold+raw.step_dt, 0.)
            landing_stop |= supported_platform & stop_on_landing''')
    s = replace(s, '                phase_weight=zeros, phase_ready=zeros.bool(), phase_stable_s=zeros,', '''                phase_weight=phase['weight'], phase_ready=phase['ready'], phase_stable_s=phase['stable_s'],
                world_velocity=data.root_lin_vel_w,
                stair_edge_window=edge_window, stair_platform_hull=on_platform & phase['stairs'],
                stair_platform_supported=supported_platform,''')
    s = replace(s, '                              ((~pit.active) & (~flat) & (dx >= 4.)))', '''                              ((~flat) & (platform_hold >= 2.)) |
                              ((~flat) & (data.root_pos_w[:, 0]-origins[:, 0] >= 4.1)))''')
    s = replace(s, '                physical = bool(guard.violation[case_id] | pit.body_contact[case_id])', '''                physical = bool(raw.termination_manager.terminated[case_id])
                if physical:
                    end_reason = 'physical_termination'
                elif bool(raw.termination_manager.time_outs[case_id]):
                    end_reason = 'timeout'
                elif bool(platform_hold[case_id] >= 2.):
                    end_reason = 'supported_platform_dwell'
                elif not bool(flat[case_id]):
                    end_reason = 'safe_tile_endpoint'
                else:
                    end_reason = 'time_budget' ''')
    s = replace(s, '                    end_step=step, seconds=elapsed, physical=physical, terms=terms,', '''                    end_step=step, seconds=elapsed, physical=physical, terms=terms,
                    end_reason=end_reason, platform_hold_s=float(platform_hold[case_id]),
                    supported_platform_reached=bool(platform_seen[case_id]),
                    actuator_or_base_contact=bool(guard.violation[case_id] | pit.body_contact[case_id]),''')
    s = replace(s, '                        physical=False, terms=["budget"], passed_4m=False)', '''                        physical=False, terms=["budget"], passed_4m=False,
                        end_reason='smoke_budget' if args.smoke else 'time_budget',
                        platform_hold_s=float(platform_hold[case['id']]),
                        supported_platform_reached=bool(platform_seen[case['id']]))''')
    s = replace(s, 'evaluation_revision="early_ablation_v15_corrected_20260922"', 'evaluation_revision="speed_brake_v20_focus_20260922"')
    s = replace(s, 'protocol_id="flat25_80m" if args.suite == "brake25" else "matched_nine_terrain_83"', 'protocol_id=args.suite')
    s = replace(s, '"20 all-direction flat cases per policy on 80 m patches, 25 s budget."', '"36 all-direction speed/braking cases per policy on 80m patches, 25s M-S-reverse-S-M."')
    s = replace(s, '"83 cases per policy across nine terrains; ordinary flat budget is 5 s and "\n                     "non-flat budget is 25 s with existing route/success termination retained."', '"Stairs use a dedicated capped outer landing; retain real phase and full collision-hull entry, "\n                     "end after supported landing dwell, safe tile endpoint, or physical/25s termination."')
    (HERE/'evaluate_focus.py').write_text(s, encoding='utf-8')


if __name__ == '__main__':
    main()
