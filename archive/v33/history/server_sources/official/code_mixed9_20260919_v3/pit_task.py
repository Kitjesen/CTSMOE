"""Independent forward pit-exit task with complete-body platform acceptance."""
import math

import torch
from isaaclab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from robot_lab.tasks.go2.mdp.commands import Go2RLGymCommand

from pit_geometry import EDGE_OFFSET, LEVEL_COUNT, SIZE, height_for_level, pit_terrain
from pit_logic import PitCourse, stable_success, top_support
from reset_geometry import CollisionGeometry
from stair_speed_logic import sample_speeds

PROFILE = "pit_exit_v1"
STAT_NAMES = ("episodes", "success", "physical", "base_contact", "overspeed", "position", "torque",
              "out_of_lane", "timeout", "front_supported", "all_supported", "early_failure")


class PitCommand(Go2RLGymCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.course = PitCourse(max_level=env.cfg.pit_max_level)
        self.course.frontier = min(self.course.frontier, self.course.max_level)
        self.fixed_levels = None
        self.learner = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.speed_bins = torch.zeros_like(self.learner)
        self.held_speed = torch.zeros(self.num_envs, device=self.device)
        self.hold = torch.zeros_like(self.held_speed)
        self.success = torch.zeros_like(self.held_speed, dtype=torch.bool)
        self.physical = self.success.clone()
        self.body_contact = self.success.clone()
        self.outside = self.success.clone()
        self.front_seen = self.success.clone()
        self.all_seen = self.success.clone()
        self.started_in_pit = self.success.clone()
        self.stop_on_platform = self.success.clone()
        self.progress = torch.zeros_like(self.held_speed)
        self.last_potential = torch.zeros_like(self.held_speed)
        self.supported = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_refresh = -1
        self.totals = torch.zeros((2, LEVEL_COUNT, 4, len(STAT_NAMES)), dtype=torch.float64, device=self.device)
        self.geometry = CollisionGeometry(env.cfg.scene.robot.spawn.asset_path, self.robot.joint_names,
                                          self.device, spacing=.04)
        self.wheel_ids, _ = self.robot.find_bodies([f"{s}_foot_Link" for s in ("fr", "fl", "rr", "rl")],
                                                  preserve_order=True)
        self.contact = env.scene["contact_forces"]
        self.ground_contacts = [env.scene[f"pit_contact_{s}"] for s in ("fr", "fl", "rr", "rl")]
        self.base_contact_ids, _ = self.contact.find_bodies("base_link")

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.hold[ids] = 0
        for value in (self.success, self.front_seen, self.all_seen, self.stop_on_platform):
            value[ids] = False
        self.started_in_pit[ids] = self._env.pit_valid_spawn[ids]
        self.progress[ids] = 0
        relative_x = self.robot.data.root_pos_w[ids, 0] - self._env.scene.env_origins[ids, 0]
        self.last_potential[ids] = relative_x.clamp(-.5, EDGE_OFFSET+.7)
        return super().reset(ids)

    def _resample(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        bins, speeds = sample_speeds(len(ids), self.device)
        self.speed_bins[ids], self.held_speed[ids] = bins, speeds
        self.commands[ids] = 0
        self.commands[ids, 0] = speeds
        self.time_left[ids] = 1e9

    def _update_command(self):
        super()._update_command()
        # Explicit external task protocol: forward until fully on the platform,
        # then a stop command (as an operator would send). This baseline does not
        # claim that a proprioceptive student autonomously detects the platform.
        self.commands[:, 0] = torch.where(self.stop_on_platform, 0., self.held_speed)
        self.commands[:, 1:] = 0

    def refresh(self):
        env = self._env
        if self.last_refresh == env.common_step_counter:
            return
        self.last_refresh = env.common_step_counter
        data = self.robot.data
        origins = env.scene.env_origins
        edge = origins[:, 0] + EDGE_OFFSET
        top = origins[:, 2] + .15 + .05 * env.scene.terrain.terrain_levels
        wheel_pos = data.body_pos_w[:, self.wheel_ids]
        force = torch.stack([sensor.data.force_matrix_w[:, 0, 0] for sensor in self.ground_contacts], dim=1)
        self.supported = top_support(wheel_pos, force, edge, top) & self.active[:, None]
        self.front_seen |= self.supported[:, :2].all(-1)
        self.all_seen |= self.supported.all(-1)
        hist = self.contact.data.net_forces_w_history[:, :, self.base_contact_ids]
        self.body_contact = torch.linalg.vector_norm(hist, dim=-1).amax(dim=(1, 2)) > 1.
        rel = data.root_pos_w - origins
        self.outside = ((rel[:, 1].abs() > 1.) | (rel[:, 0] < -.8) | (rel[:, 0] > 4.)) & self.active
        self.physical = (env.actuator_guard.violation | self.body_contact) & self.active
        body_clear = torch.zeros_like(self.success)
        candidate = self.supported.all(-1) & (rel[:, 0] > EDGE_OFFSET+.45)
        ids = candidate.nonzero().flatten()
        if len(ids):
            points = self.geometry.world_points(data.joint_pos[ids], data.root_state_w[ids, :7])
            body_clear[ids] = ((points[..., 0].amin(1) > edge[ids]+.06)
                               & (points[..., 1].amin(1) > origins[ids, 1]-1.5)
                               & (points[..., 1].amax(1) < origins[ids, 1]+1.5))
        self.stop_on_platform |= body_clear & ~self.physical
        upright = -data.projected_gravity_b[:, 2] > math.cos(math.radians(15))
        quiet = (torch.linalg.vector_norm(data.root_lin_vel_w[:, :2], dim=-1) < .15)
        quiet &= torch.linalg.vector_norm(data.root_ang_vel_w, dim=-1) < .3
        self.hold, self.success = stable_success(
            self.hold, self.supported, body_clear, upright, quiet,
            self.physical | self.outside, self.started_in_pit, env.step_dt)
        potential = rel[:, 0].clamp(-.5, EDGE_OFFSET+.7)
        self.progress = ((potential-self.last_potential)/env.step_dt).clamp(-5., 5.) * self.active
        self.last_potential.copy_(potential)

    def record_and_resample_levels(self, ids):
        env = self._env
        ids = ids[env.episode_length_buf[ids] > 0]
        if not len(ids):
            return
        self.refresh()
        levels = env.scene.terrain.terrain_levels[ids]
        key = (self.learner[ids]*LEVEL_COUNT + levels)*4 + self.speed_bins[ids]
        causes = env.actuator_guard.violation_causes
        timeout = env.episode_length_buf[ids] >= env.max_episode_length-1
        values = torch.stack((torch.ones_like(self.hold[ids]), self.success[ids], self.physical[ids],
                              self.body_contact[ids], causes["speed"][ids], causes["position"][ids],
                              causes["torque"][ids], self.outside[ids], timeout & ~self.success[ids],
                              self.front_seen[ids], self.all_seen[ids],
                              self.physical[ids] & (env.episode_length_buf[ids]*env.step_dt < .35)), -1)
        self.totals.view(-1, len(STAT_NAMES)).index_add_(0, key, values.double())
        frontier = ids[(self.learner[ids] == 1) & (levels == self.course.frontier)]
        if self.fixed_levels is None and len(frontier):
            self.course.observe(self.speed_bins[frontier].cpu().tolist(), self.success[frontier].cpu().tolist(),
                                self.physical[frontier].cpu().tolist())
        if self.fixed_levels is None:
            sampled = torch.randint(self.course.frontier+1, (len(ids),), device=self.device)
            sampled = torch.where(torch.rand(len(ids), device=self.device) < .8, self.course.frontier, sampled)
        else:
            sampled = self.fixed_levels[ids]
        terrain = env.scene.terrain
        terrain.terrain_levels[ids] = sampled
        terrain.env_origins[ids] = terrain.terrain_origins[sampled, terrain.terrain_types[ids]]
        env.scene.env_origins[ids] = terrain.env_origins[ids]

    def snapshot(self):
        totals = self.totals.detach().cpu()
        course_log = {"frontier": self.course.frontier, "max_level": self.course.max_level,
                      "min_samples": self.course.min_samples, "streak": self.course.streak,
                      "window": self.course.window, "decision_count": len(self.course.decisions),
                      "decisions": self.course.decisions[-5:]}
        result = {"profile": PROFILE, "frontier_level": self.course.frontier,
                  "frontier_height_m": height_for_level(self.course.frontier),
                  "max_height_m": height_for_level(self.course.max_level),
                  "course": course_log, "by_learner": {},
                  "acceptance": "all wheels supported on top, full collision body past edge, quiet/upright for 2s",
                  "stop_protocol": "external perfect-state stop command on platform; not autonomous student detection",
                  "course_evidence": "disjoint stochastic training batches; not independent held-out evaluation"}
        for learner, name in enumerate(("teacher", "student")):
            result["by_learner"][name] = {
                "totals": dict(zip(STAT_NAMES, totals[learner].sum((0, 1)).long().tolist(), strict=True)),
                "by_height_speed": totals[learner].long().tolist(),
            }
        return result


def reset_pit_root(env, env_ids):
    command = pit_command(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    robot = env.scene["robot"]
    pose = robot.data.default_root_state[ids, :7].clone()
    origins = env.scene.env_origins[ids]
    pose[:, :3] += origins
    pose[:, 0] += torch.empty(len(ids), device=env.device).uniform_(-.25, .25)
    pose[:, 1] += torch.empty(len(ids), device=env.device).uniform_(-.1, .1)
    yaw = torch.empty(len(ids), device=env.device).uniform_(-.04, .04)
    pose[:, 3:] = 0
    pose[:, 3], pose[:, 6] = torch.cos(yaw/2), torch.sin(yaw/2)
    points = command.geometry.world_points(robot.data.joint_pos[ids], pose)
    pose[:, 2] += origins[:, 2] + .03 - points[..., 2].amin(1)
    if not hasattr(env, "pit_valid_spawn"):
        env.pit_valid_spawn = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.pit_valid_spawn[ids] = (points[..., 0].amax(1) < origins[:, 0]+EDGE_OFFSET-.1)
    if not bool(env.pit_valid_spawn[ids].all()):
        raise RuntimeError("Pit reset intersects front wall")
    robot.write_root_pose_to_sim(pose, ids)
    robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), ids)


def pit_success(env):
    command = pit_command(env)
    command.refresh()
    return command.success


def pit_outside(env):
    command = pit_command(env)
    command.refresh()
    return command.outside


def pit_reward(env, kind):
    c = pit_command(env)
    c.refresh()
    if kind == "progress":
        return c.progress
    if kind == "success":
        return c.success.float()/env.step_dt
    if kind == "stationary":
        return (c.stop_on_platform & ~c.physical).float() * torch.exp(
            -torch.sum(c.robot.data.root_lin_vel_w[:, :2].square(), -1)/.04)
    raise ValueError(kind)


def pit_command(env):
    """Resolve the pit state owned by either a dedicated or a mixed command term."""
    command = env.command_manager.get_term("base_velocity")
    return command.pit if hasattr(command, "pit") else command


def pit_curriculum(env, env_ids):
    c = env.command_manager.get_term("base_velocity")
    c.record_and_resample_levels(torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    return float(c.course.frontier)


def configure_pit(cfg, max_level=4):
    cfg.pit_profile = PROFILE
    cfg.pit_max_level = max_level
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=SIZE, num_rows=LEVEL_COUNT, num_cols=4, curriculum=True, difficulty_range=(0., 1.),
        use_cache=False, border_width=2., sub_terrains={"pit": SubTerrainBaseCfg(function=pit_terrain, proportion=1.)})
    cfg.scene.terrain.max_init_terrain_level = min(2, max_level)
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    for side in ("fr", "fl", "rr", "rl"):
        setattr(cfg.scene, f"pit_contact_{side}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_foot_Link", update_period=0., history_length=1,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"]))
    cfg.commands.base_velocity.class_type = PitCommand
    cfg.commands.base_velocity.terrain_max_command_ranges = {
        "pit": {"lin_vel_x": (.2, 1.), "lin_vel_y": (0., 0.), "ang_vel_yaw": (0., 0.)}}
    cfg.commands.base_velocity.ranges.lin_vel_x = (.2, 1.)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
    cfg.commands.base_velocity.ranges.ang_vel_yaw = (0., 0.)
    cfg.commands.base_velocity.command_range_curriculum = []
    cfg.commands.base_velocity.zero_command_curriculum = None
    cfg.events.reset_base.func = reset_pit_root
    cfg.events.reset_base.params = {}
    cfg.events.reset_robot_joints.params["position_range"] = (.95, 1.05)
    cfg.events.randomize_push_robot = None
    cfg.events.randomize_rigid_body_material.params.update(
        static_friction_range=(.6, 1.2), dynamic_friction_range=(.6, 1.2), restitution_range=(0., .05))
    cfg.curriculum.terrain_levels = CurriculumTermCfg(func=pit_curriculum)
    cfg.curriculum.base_height_l2 = None
    cfg.curriculum.base_linear_velocity = None
    for name in ("base_height_l2", "lin_vel_z_l2", "feet_regulation"):
        getattr(cfg.rewards, name).weight = 0.
    cfg.rewards.pit_progress = RewardTermCfg(func=pit_reward, weight=3., params={"kind": "progress"})
    cfg.rewards.pit_success = RewardTermCfg(func=pit_reward, weight=10., params={"kind": "success"})
    cfg.rewards.pit_stationary = RewardTermCfg(func=pit_reward, weight=2., params={"kind": "stationary"})
    cfg.terminations.pit_success = TerminationTermCfg(func=pit_success, time_out=False)
    cfg.terminations.pit_outside = TerminationTermCfg(func=pit_outside, time_out=False)
    cfg.episode_length_s = 25.
