"""Produce a reviewable isolated evaluator from the corrected, frozen source."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "early_ablation_v15_20260921/evaluation_audit_20260922/evaluate_corrected.py"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Evaluator source differs at: {old[:80]}")
    return text.replace(old, new)


def build():
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_once(text, "    cases = build_cases(selection, args.suite)",
                        "    from runup_protocol import build_cases as build_runup_cases\n    cases = build_runup_cases(selection, args.suite)")
    text = replace_once(text, "        env.stair_axis[other_ids] = 0", '''        env.stair_axis[other_ids] = 0
        from slope_runup import place
        gaps = torch.tensor([cases[i]["gap"] for i in env_ids.tolist()], device=env.device)
        lateral = torch.tensor([cases[i]["lateral"] for i in env_ids.tolist()], device=env.device)
        place(env, env_ids, gaps, torch.zeros_like(env_ids), lateral=lateral)''')
    text = replace_once(text, '    observations = env.get_observations()', '''    entry_seen = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    entry_speeds = torch.full((raw.num_envs,), float("nan"), device=raw.device)
    substep_fields = ("joint_velocity", "joint_target_velocity", "applied_torque", "computed_torque", "root_state", "wheel_forces", "violation")
    substep_trace = {key: [] for key in substep_fields}
    # All physics samples are retained: a 20ms end-state can hide a 5ms spike.
    instrumented_measure = guard.measure
    def detailed_measure():
        instrumented_measure()
        values = (robot.data.joint_vel, robot.data.joint_vel_target,
                  robot.data.applied_torque, robot.data.computed_torque, robot.data.root_state_w,
                  torch.stack([sensor.data.force_matrix_w[:, 0, 0] for sensor in pit.ground_contacts], dim=1),
                  torch.stack(list(guard.violation_causes.values()), dim=1))
        for key, value in zip(substep_fields, values):
            substep_trace[key].append(value.clone())
    guard.measure = detailed_measure
    observations = env.get_observations()''')
    text = replace_once(text, '            dx = data.root_pos_w[:, 0] - initial[:, 0]', '''            # Same absolute exit for short and long run-up; displacement from spawn would bias the test.
            dx = data.root_pos_w[:, 0] - origins[:, 0]
            pending = (active & ~entry_seen).nonzero().flatten()
            front = torch.full((raw.num_envs,), -float("inf"), device=raw.device)
            if len(pending):
                points = pit.geometry.world_points(data.joint_pos[pending], data.root_state_w[pending, :7])
                front[pending] = points[..., 0].amax(-1) - origins[pending, 0]
            new_entry = active & ~entry_seen & (front >= 1.5)
            entry_speeds[new_entry] = data.root_lin_vel_w[new_entry, 0]
            entry_seen |= new_entry''')
    text = replace_once(text, '                    climb=bool(geom_seen[case_id] | pit.success[case_id]),', '''                    entry_speed_mps=float(entry_speeds[case_id]) if entry_seen[case_id] else None,
                    climb=bool(geom_seen[case_id] | pit.success[case_id]),''')
    text = replace_once(text, '    base_id = robot.body_names.index("base_link")', '''    np.savez_compressed(args.output / "physics_5ms.npz",
                        **{key: torch.stack(value).cpu().numpy() for key, value in substep_trace.items()})
    base_id = robot.body_names.index("base_link")''')
    text = replace_once(text, '        evaluation_revision="early_ablation_v15_corrected_20260922",', '''        runup_receipt=raw.slope_runup_placement_receipt,
        physics_sample_dt=raw.physics_dt,
        evaluation_revision="uphill_runup_v16_20260922",
        fixed_case_diagnostic=True,''')
    text = replace_once(text, '    print("DONE", len(cases), flush=True)', '''    from runup_protocol import summarize
    (args.output / "runup_summary.json").write_text(json.dumps(summarize(manifest), indent=2), encoding="utf-8")
    print("DONE", len(cases), flush=True)''')
    # Imports retained from the frozen evaluator preserve its source contract checks.
    text = text.replace('from early_protocol import (build_cases, checkpoint_metadata, load_protocol,',
                        'from early_protocol import (checkpoint_metadata, load_protocol,')
    text = text.replace('    initial = robot.data.root_pos_w.clone()\n', '')
    text = text.replace('((~pit.active) & (~flat) & (dx >= 4.)))',
                        '((~pit.active) & (~flat) & (dx >= 4.)) | (elapsed >= 25.))')
    text = text.replace('protocol_id="flat25_80m" if args.suite == "flat25" else "matched_nine_terrain_83",',
                        'protocol_id="uphill_180_matched_speed_front_gap",')
    text = text.replace('"83 cases per policy across nine terrains; ordinary flat budget is 5 s and "\n                     "non-flat budget is 25 s with existing route/success termination retained."',
                        '"90 uphill cases per policy, 25s maximum, common x=4m exit; 35/40/45 degrees, "\n                     "five speeds, two front gaps and three lateral offsets; 180 total for student/teacher."')
    path = HERE / "evaluate_runup.py"
    path.write_text(text, encoding="utf-8")
    # Exact old-batch replay retains its original case ordering and reset geometry.
    replay = SOURCE.read_text(encoding="utf-8")
    begin = text.index('    substep_fields =')
    end = text.index('    observations = env.get_observations()', begin)
    replay = replace_once(replay, '    observations = env.get_observations()',
                          text[begin:end] + '    observations = env.get_observations()')
    save_begin = text.index('    np.savez_compressed(args.output / "physics_5ms.npz"')
    save_end = text.index('    base_id =', save_begin)
    replay = replace_once(replay, '    base_id = robot.body_names.index("base_link")',
                          text[save_begin:save_end] + '    base_id = robot.body_names.index("base_link")')
    (HERE / 'evaluate_physics.py').write_text(replay, encoding='utf-8')
    return path


if __name__ == "__main__":
    print(build())
