"""Adapt preserved common evaluations only for the explicit V22 hip contract."""
import hashlib
import json
from pathlib import Path

from build_candidate import replace_once

HERE = Path(__file__).resolve().parent
OUT = HERE / 'assessment'
OUT.mkdir(exist_ok=False)
AUDIT = '/root/autodl-tmp/cts-omni-repair-v19-20260922/speed_brake_v20/landing_audit_v21'

for source, target in (('evaluate_measured_reference.py', 'evaluate_hip.py'),
                       ('evaluate_local_platform_reference.py', 'evaluate_hip_platform.py')):
    text = (HERE/source).read_text(encoding='utf-8')
    text = replace_once(text, '    return parser.parse_args()',
        '    parser.add_argument("--hip-profile", choices=("v21", "v22"), default="v22")\n'
        '    parser.add_argument("--stairs15", action="store_true")\n'
        '    return parser.parse_args()')
    text = replace_once(text, '    cfg.terminations.startup_window = None\n',
        '    hip_contract = None\n'
        '    if args.hip_profile == "v22":\n'
        '        from hip_guard import configure as configure_hip\n'
        '        from hip_contract import contract\n'
        '        configure_hip(cfg)\n'
        '        hip_contract = contract()\n'
        '    if args.stairs15:\n'
        '        for name in ("stairs_up", "stairs_down"):\n'
        '            cfg.scene.terrain.terrain_generator.sub_terrains[name].step_height_range = (.15, .15)\n'
        '    cfg.terminations.startup_window = None\n')
    text = replace_once(text, '    manifest = dict(\n',
        '    manifest = dict(hip_contract=hip_contract, stairs15=args.stairs15,\n')
    # Diagnostic-only per-physics-step peaks; no reward, reset or contact mutation.
    text = replace_once(text, '"hard_excess", "saturation", "gravity",',
        '"hard_excess", "saturation", "acceleration_peak", "calf_contact_peak", "gravity",')
    text = replace_once(text, 'for key in ("torque_peak", "speed_peak", "hard_excess", "saturation")}',
        'for key in ("torque_peak", "speed_peak", "hard_excess", "saturation", "acceleration_peak")}\n'
        '    calf_ids = [i for i,n in enumerate(pit.contact.body_names) if n.endswith("_calf_Link")]\n'
        '    assert len(calf_ids) == 4\n'
        '    step_stats["calf_contact_peak"] = torch.zeros(raw.num_envs, len(calf_ids), device=raw.device)')
    text = replace_once(text, '        terminal_targets.copy_(data.joint_pos_target)\n',
        '        terminal_targets.copy_(data.joint_pos_target)\n'
        '        step_stats["acceleration_peak"].copy_(torch.maximum(step_stats["acceleration_peak"], data.joint_acc.abs()))\n'
        '        contact = pit.contact.data.net_forces_w[:, calf_ids].norm(dim=-1)\n'
        '        step_stats["calf_contact_peak"].copy_(torch.maximum(step_stats["calf_contact_peak"], contact))\n')
    if target.endswith('platform.py'):
        text = replace_once(text, "sys.path.insert(0, '/root/autodl-tmp/cts-v15-evaluation-audit-20260922')",
            f"sys.path.insert(0, '{AUDIT}')\nsys.path.insert(0, '/root/autodl-tmp/cts-v15-evaluation-audit-20260922')")
    (OUT/target).write_text(text, encoding='utf-8')
sources = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
           for name in ('evaluate_measured_reference.py','evaluate_local_platform_reference.py')}
(OUT/'SOURCE_EVALUATORS.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
