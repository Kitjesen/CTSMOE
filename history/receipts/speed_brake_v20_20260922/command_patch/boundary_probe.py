"""Reproducible CPU geometry audit; this is not a simulator qualification."""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parents[1] / 'omni_repair_v19_20260922' / 'recovery01'
sys.path.append(str(BASELINE / 'code'))

import flat_boundary  # noqa: E402
import omni_commands  # noqa: E402


def _baseline_sampler():
    spec = importlib.util.spec_from_file_location('frozen_v19_omni', BASELINE / 'code' / 'omni_commands.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_command_snapshot(value):
    if isinstance(value, dict):
        if 'omni_commands' in value:
            return value['omni_commands']
        for child in value.values():
            result = _find_command_snapshot(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_command_snapshot(child)
            if result is not None:
                return result
    return None


def audit_positions(sampler, positions, yaw, lower, upper, seed):
    generator = torch.Generator().manual_seed(seed)
    original, kinds = sampler.sample_goals(len(positions), generator=generator, moving_only=True)
    result, reflected, resolved = flat_boundary.reflect_goals(positions, yaw, original, lower, upper)
    first_unresolved = int((~resolved).sum())
    reflections = int((reflected & resolved).sum())
    attempts = 0
    for _ in range(4):
        missing = (~resolved).nonzero().flatten()
        if not len(missing):
            break
        attempts += len(missing)
        retry, retry_kinds = sampler.sample_goals(len(missing), generator=generator, moving_only=True)
        retry, changed, accepted = flat_boundary.reflect_goals(
            positions[missing], yaw[missing], retry, lower, upper)
        selected = missing[accepted]
        result[selected] = retry[accepted]
        kinds[selected] = retry_kinds[accepted]
        resolved[selected] = True
        reflections += int((changed & accepted).sum())
    missing = (~resolved).nonzero().flatten()
    if len(missing):
        recovery, recovery_kinds = flat_boundary.cardinal_recovery(
            positions[missing], yaw[missing], lower, upper)
        result[missing], kinds[missing] = recovery, recovery_kinds
    kinds = omni_commands.truthful_kinds(result, kinds)
    trajectory = flat_boundary.paths(positions, yaw, result)
    valid = ((trajectory >= lower - 1e-5) & (trajectory <= upper + 1e-5)).all(dim=(1, 2))
    assert bool(valid.all())
    assert bool((result.abs().sum(-1) > 0).all())
    return dict(
        samples=len(positions),
        first_goals_unresolved=first_unresolved,
        reflection_acceptances=reflections,
        replacement_draws=attempts,
        cardinal_recoveries=len(missing),
        magnitude_vector_changed=int((~torch.isclose(original.abs(), result.abs()).all(-1)).sum()),
        final_kind_counts=dict(zip(sampler.NAMES, torch.bincount(kinds, minlength=len(sampler.NAMES)).tolist())),
        final_nonzero_abs_ranges=omni_commands.observed_nonzero_abs_ranges(result),
        all_final_sampled_paths_within_planning_bounds=bool(valid.all()),
    )


def main():
    torch.set_num_threads(1)
    baseline = _baseline_sampler()
    source = BASELINE / 'PREFLIGHT_RUN.json'
    receipt = json.loads(source.read_text(encoding='utf-8'))
    boundary = _find_command_snapshot(receipt)['boundary']
    lower = torch.tensor(boundary['physical_lower_xy']) + boundary['planning_margin_m']
    upper = torch.tensor(boundary['physical_upper_xy']) - boundary['planning_margin_m']
    count = 6000
    generator = torch.Generator().manual_seed(20260922)
    positions = lower + (upper-lower) * torch.rand(count, 2, generator=generator)
    yaw = (torch.rand(count, generator=generator) * 2 - 1) * math.pi
    edge = positions.clone()
    sides = torch.randint(0, 4, (count,), generator=generator)
    for side in range(4):
        axis = side // 2
        edge[sides == side, axis] = lower[axis] + .01 if side % 2 == 0 else upper[axis] - .01
    corners = torch.where(torch.rand(count, 2, generator=generator) < .5, lower + .01, upper - .01)
    cases = {}
    for name, points in [('uniform_planning_rectangle', positions), ('one_cm_inside_planning_edge', edge),
                         ('one_cm_inside_planning_corner', corners)]:
        cases[name] = {
            'v19': audit_positions(baseline, points, yaw, lower, upper, 190922),
            'v20': audit_positions(omni_commands, points, yaw, lower, upper, 190922),
        }
    report = dict(
        scope='CPU body-twist geometry stress audit; no physics, training, or held-out policy performance claim',
        bounds_source=str(source),
        bounds_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        baseline_omni_sha256=hashlib.sha256((BASELINE / 'code' / 'omni_commands.py').read_bytes()).hexdigest(),
        patched_omni_sha256=hashlib.sha256((HERE / 'omni_commands.py').read_bytes()).hexdigest(),
        sampling='moving-only stress draws; identical position/yaw and sampler seeds for V19/V20; 4 retries',
        flat_override=omni_commands.sampling_contract(),
        inherited_global_ranges_in_frozen_receipt=receipt['command_ranges'],
        physical_lower_xy=boundary['physical_lower_xy'],
        physical_upper_xy=boundary['physical_upper_xy'],
        planning_lower_xy=lower.tolist(), planning_upper_xy=upper.tolist(),
        planning_size_xy=(upper-lower).tolist(),
        max_straight_longitudinal_5s_m=5. * omni_commands.NONZERO_ABS_RANGES['lin_vel_x'][1],
        max_diagonal_5s_path_length_m=5. * math.hypot(2., 1.),
        cases=cases,
    )
    output = HERE / 'BOUNDARY_PROBE.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: {version: {key: item[key] for key in (
        'first_goals_unresolved', 'replacement_draws', 'cardinal_recoveries', 'magnitude_vector_changed')}
        for version, item in value.items()} for name, value in cases.items()}, indent=2))
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
