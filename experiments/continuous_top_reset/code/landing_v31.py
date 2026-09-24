"""Control-step landing events with 200 Hz force-history peak evidence.

Downward velocity is the previous 50 Hz wheel-link velocity, not a measured
impact-time normal velocity. Initial spawn/first contact is excluded until an
environment has supported itself; subsequent landings on all terrains count.
"""
import torch

PROFILE = 'landing_force_velocity_v31'
FORCE_WEIGHT = -.5
VELOCITY_WEIGHT = -1.
FORCE_MULTIPLE_QUARTER_WEIGHT = 3.
VELOCITY_FREE_MPS = .25


def penalties(force, downward_before, landing, recent, body_weight):
    reference = body_weight[:, None] / 4
    force_excess = (force / reference - FORCE_MULTIPLE_QUARTER_WEIGHT).clamp_min(0)
    force_cost = (force_excess.square() * recent).mean(-1)
    speed_cost = ((downward_before - VELOCITY_FREE_MPS).clamp_min(0).square() * landing).sum(-1)
    return force_cost, speed_cost


class LandingState:
    def __init__(self, env):
        self.env = env
        robot = env.scene['robot']
        names = [f'{s}_foot_Link' for s in ('fr', 'fl', 'rr', 'rl')]
        self.body_ids = [robot.body_names.index(n) for n in names]
        sensor = env.scene['landing_contacts']
        self.sensor_ids = [sensor.body_names.index(n) for n in names]
        if sensor.data.net_forces_w_history.shape[1] < 4:
            raise RuntimeError('Need all four 200 Hz force samples per control step')
        self.body_weight = robot.root_physx_view.get_masses().to(env.device).sum(-1) * 9.81
        shape = (env.num_envs, 4)
        self.previous_contact = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.supported = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.air = torch.zeros(shape, device=env.device)
        self.age = torch.full(shape, 1., device=env.device)
        self.previous_down = torch.zeros(shape, device=env.device)
        self.force_cost = torch.zeros(env.num_envs, device=env.device)
        self.speed_cost = torch.zeros_like(self.force_cost)
        self.last_clock = -1
        self.landing_count = torch.zeros(4, dtype=torch.long, device=env.device)
        self.force_peak = torch.zeros(4, device=env.device)
        self.down_peak = torch.zeros(4, device=env.device)
        self.force_cost_total = torch.zeros((), device=env.device)
        self.speed_cost_total = torch.zeros((), device=env.device)

    def update(self):
        e = self.env
        if self.last_clock == e.common_step_counter:
            return
        self.last_clock = e.common_step_counter
        forces = e.scene['landing_contacts'].data.net_forces_w_history[:, :, self.sensor_ids, 2].clamp_min(0)
        peak = forces.amax(1)
        current = e.scene['landing_contacts'].data.net_forces_w[:, self.sensor_ids, 2] > 5.
        reset = e.episode_length_buf <= 1
        self.supported[reset] = False
        self.air[reset] = 0.
        self.age[reset] = 1.
        self.previous_contact[reset] = current[reset]
        self.previous_down[reset] = 0.
        # Require a short flight to avoid counting contact-force chatter as a
        # series of new impacts. A brief contact within the physics history counts.
        landing = (peak > 5.) & ~self.previous_contact & (self.air >= .04) & self.supported[:, None] & ~reset[:, None]
        self.age = torch.where(landing, 0., self.age + e.step_dt)
        recent = (self.age <= .1) & ~reset[:, None]
        self.force_cost, self.speed_cost = penalties(peak, self.previous_down, landing, recent, self.body_weight)
        self.force_peak = torch.maximum(self.force_peak, (peak * recent).amax(0))
        self.down_peak = torch.maximum(self.down_peak, (self.previous_down * landing).amax(0))
        self.landing_count += landing.sum(0)
        self.force_cost_total += self.force_cost.sum()
        self.speed_cost_total += self.speed_cost.sum()
        self.air = torch.where(current | landing, 0., self.air + e.step_dt)
        self.supported |= current.any(-1) & ~reset
        self.previous_contact = current.clone()
        self.previous_down = (-e.scene['robot'].data.body_lin_vel_w[:, self.body_ids, 2]).clamp_min(0).clone()


def state(env):
    if not hasattr(env, 'landing_v31'):
        env.landing_v31 = LandingState(env)
    env.landing_v31.update()
    return env.landing_v31


def reward(env, kind):
    value = state(env)
    return value.force_cost if kind == 'force' else value.speed_cost


def snapshot(env):
    s = state(env)
    return dict(profile=PROFILE, force_weight=FORCE_WEIGHT, velocity_weight=VELOCITY_WEIGHT,
        threshold_N=(s.body_weight * .75).mean().item(), previous_frame_speed_free_mps=VELOCITY_FREE_MPS,
        landings=s.landing_count.cpu().tolist(), force_peak_N=s.force_peak.cpu().tolist(),
        previous_frame_down_peak_mps=s.down_peak.cpu().tolist(),
        force_cost_sum=s.force_cost_total.item(), speed_cost_sum=s.speed_cost_total.item(),
        scope='all terrains; excludes initial unsupported settling; cumulative training evidence',
        force_history_hz=200, velocity_hz=50, threshold_is_hardware_limit=False)


def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensorCfg
    # The original sensor history also feeds the critic. A dedicated sensor
    # preserves its 291-dimensional checkpoint contract while covering all
    # four physics steps for landing rewards.
    cfg.scene.landing_contacts = ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/.*_foot_Link', history_length=4,
        update_period=cfg.sim.dt, track_air_time=False)
    cfg.rewards.landing_force_v31 = RewardTermCfg(func=reward, weight=FORCE_WEIGHT, params={'kind': 'force'})
    cfg.rewards.landing_velocity_v31 = RewardTermCfg(func=reward, weight=VELOCITY_WEIGHT, params={'kind': 'velocity'})
