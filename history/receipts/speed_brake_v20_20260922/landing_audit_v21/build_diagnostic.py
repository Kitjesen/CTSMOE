"""Build an isolated evaluator; preserve the published V20 evaluator unchanged."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE.parent / 'evaluate_focus.py').read_text(encoding='utf-8')


def replace(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError(f'Expected one audited insertion: {old[:70]}')
    source = source.replace(old, new)


replace('from focus_protocol import build_focus_cases, brake_factor',
        'from focus_protocol import build_focus_cases, brake_factor\n'
        'from local_platform import local_platform_support, platform_from_probes, update_dwell, phase_ids')
replace('    platform_seen = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)',
        '    platform_seen = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)\n'
        '    platform_hold_steps = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.int64)')
replace('    original_measure = guard.measure', '''    if args.suite != 'stairend':
        raise ValueError('This isolated diagnostic revision is stairs-only')
    offsets = torch.tensor([[x, y, 0.] for x in (3.2, 3.5, 3.8) for y in (-1., 0., 1.)], device=raw.device)
    probe_start = origins[:, None, :] + offsets[None, :, :]
    probe_start[..., 2] += 10.
    probe_direction = torch.zeros_like(probe_start)
    probe_direction[..., 2] = -1.
    probe_heights = raycast_mesh(probe_start, probe_direction, mesh)[0][..., 2]
    platform_z = platform_from_probes(probe_heights)
    (args.output/'PLATFORM_GEOMETRY.json').write_text(json.dumps(dict(
        probe_offsets=offsets.cpu().tolist(), heights=probe_heights.cpu().tolist(),
        world_platform_z=platform_z.cpu().tolist(), max_probe_spread_m=.01), indent=2))
    trace_names_extra = ('local_platform_count', 'local_platform_level', 'landing_phase',
                        'local_hold_steps', 'old_global_ready', 'local_platform_z')
    traces.update({key: [] for key in trace_names_extra})
    assert abs(raw.step_dt-.02) < 1e-10
    original_measure = guard.measure''')
replace("            supported_platform = on_platform & phase['ready'] & phase['stairs']", '''            force_z = torch.stack([sensor.data.force_matrix_w[:, 0, 0, 2]
                                   for sensor in pit.ground_contacts], dim=1)
            clearance = data.body_pos_w[:, pit.wheel_ids, 2] - hits[:, 1:, 2] - .093
            local_support = local_platform_support(local, hits[:, 1:, 2], clearance, force_z, platform_z)
            torch.testing.assert_close(on_platform, local_support['hull'])
            supported_platform = local_support['supported'] & phase['stairs']''')
replace('            platform_hold = torch.where(supported_platform, platform_hold+raw.step_dt, 0.)',
        '            platform_hold_steps, dwell_done = update_dwell(platform_hold_steps, supported_platform)\n'
        '            platform_hold = platform_hold_steps * raw.step_dt\n'
        '            stage = phase_ids(edge_window, on_platform, supported_platform, dwell_done)')
replace('            values = dict(', '''            values = dict(
                local_platform_count=local_support['count'], local_platform_level=local_support['local_level'],
                landing_phase=stage, local_hold_steps=platform_hold_steps,
                old_global_ready=phase['ready'], local_platform_z=platform_z,''')
replace('((~flat) & (platform_hold >= 2.))', '((~flat) & dwell_done)')
replace('elif bool(platform_hold[case_id] >= 2.):', 'elif bool(dwell_done[case_id]):')
replace('evaluation_revision="speed_brake_v20_focus_20260922"',
        'evaluation_revision="local_platform_audit_v21_20260922"')
replace('        history=history, cases=cases,',
        "        local_platform_diagnostic=dict(full_hull=True, known_platform_height=True, min_contacts=3, dwell_s=2., whole_scan_gate=False),\n        history=history, cases=cases,")
(HERE/'evaluate_local_platform.py').write_text(source, encoding='utf-8')
(HERE/'focus_protocol.py').write_bytes((HERE.parent/'focus_protocol.py').read_bytes())
print('Built isolated evaluator; V20 source unchanged')
