"""Controlled Isaac component test, never counted as policy climbing performance."""
import torch


def validate_support(raw, env, steps=160):
    command = raw.command_manager.get_term("base_velocity")
    robot = raw.scene["robot"]
    ids = torch.arange(raw.num_envs, device=raw.device)
    cases = ids % 4  # floor, platform/valid start, platform/invalid start, straddling lip
    terrain = raw.scene.terrain
    terrain.terrain_levels[:] = 2
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    raw.scene.env_origins[:] = terrain.env_origins
    original_curriculum = raw.curriculum_manager.compute
    raw.curriculum_manager.compute = lambda env_ids: None
    raw.reset()
    q = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), ids)
    pose = robot.data.default_root_state[:, :7].clone()
    pose[:, :3] += raw.scene.env_origins
    pose[cases != 0, 2] += .25
    pose[(cases == 1) | (cases == 2), 0] += 2.4
    pose[cases == 3, 0] += 1.5
    pose[:, 2] += .025
    robot.write_root_pose_to_sim(pose, ids)
    robot.write_root_velocity_to_sim(torch.zeros(raw.num_envs, 6, device=raw.device), ids)
    raw.action_manager.reset(ids)
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    command.started_in_pit[:] = cases != 2
    command.last_refresh = -1
    successes = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)
    ever_supported = successes.clone()
    max_filtered_z = torch.zeros(4, device=raw.device)
    original_compute = raw.termination_manager.compute

    def observe_before_reset():
        result = original_compute()
        successes.logical_or_(command.success)
        ever_supported.logical_or_(command.supported.all(-1))
        forces = torch.stack([s.data.force_matrix_w[:, 0, 0, 2] for s in command.ground_contacts], -1)
        max_filtered_z.copy_(torch.maximum(max_filtered_z, forces.amax(0)))
        return result

    raw.termination_manager.compute = observe_before_reset
    try:
        for _ in range(steps):
            env.step(torch.zeros(raw.num_envs, 16, device=raw.device))
    finally:
        raw.termination_manager.compute = original_compute
        raw.curriculum_manager.compute = original_curriculum
    result = {"mode": "manually placed component test; not policy performance",
              "filter": "/World/ground/terrain/mesh", "max_filtered_z_n": max_filtered_z.cpu().tolist(),
              "cases": {}}
    for case, name in enumerate(("floor", "platform_valid_start", "platform_invalid_start", "straddling_lip")):
        mask = cases == case
        result["cases"][name] = dict(count=int(mask.sum()), success=int(successes[mask].sum()),
                                     ever_four_supported=int(ever_supported[mask].sum()))
    if not bool((max_filtered_z > 5.).all()):
        raise AssertionError(f"Terrain contact filter failed: {result}")
    if not bool(successes[cases == 1].any()) or bool(successes[cases != 1].any()):
        raise AssertionError(f"Platform success positive/negative cases failed: {result}")
    return result
