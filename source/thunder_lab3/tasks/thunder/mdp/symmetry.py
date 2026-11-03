"""Thunder-specific sagittal reflection for pinned RSL's segmented CTS batches.

The signed permutation follows this URDF's axes, including all four wheels.
It is an involution of observation/action coordinates, not proof of equal returns:
the shipped link origins, mass distribution and base inertia are not exactly
mirror-symmetric. Mirrored PPO samples therefore impose an approximate prior.
"""
import torch

from go2_rl_lab.utils.symmetry import Go2Symmetry
from .privileged import JOINTS, tensor


JOINT_PERM = (3,4,5,0,1,2,9,10,11,6,7,8,13,12,15,14)
FOOT_PERM = (1,0,3,2)
GAIN_PERM = (1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14)


def reflected_ray_indices(starts, directions=None, tolerance=1e-6):
    """Resolve the actual local scan coordinates rather than assuming grid ordering."""
    if starts.ndim != 2 or starts.shape[1] != 3 or not torch.isfinite(starts).all():
        raise ValueError("Expected finite local ray starts [rays,3]")
    reflected = starts*starts.new_tensor([1.,-1.,1.])
    distance = torch.cdist(reflected, starts, compute_mode="donot_use_mm_for_euclid_dist")
    error, indices = distance.min(dim=1)
    if (error > tolerance).any() or indices.unique().numel() != len(starts):
        raise ValueError("Height scan is not bijectively symmetric")
    if not torch.equal(indices[indices], torch.arange(len(starts), device=starts.device)):
        raise ValueError("Height scan reflection is not an involution")
    if directions is not None:
        if directions.shape != starts.shape or not torch.isfinite(directions).all():
            raise ValueError("Expected finite directions matching local ray starts")
        if not torch.allclose(directions[indices], directions*directions.new_tensor([1.,-1.,1.]), atol=tolerance, rtol=0.):
            raise ValueError("Height ray directions are not sagittally symmetric")
    return indices


class ThunderSymmetryMapper:
    def __init__(self, env):
        base = getattr(env, "unwrapped", env)
        robot = base.scene["robot"]
        ids = [robot.joint_names.index(name) for name in JOINTS]
        defaults = tensor(robot.data.default_joint_pos)[..., ids[:12]]
        if not torch.allclose(defaults, -defaults[...,list(JOINT_PERM[:12])], atol=1e-7, rtol=0.):
            raise ValueError("Signed Thunder symmetry requires mirrored default leg angles")
        sensor = base.scene["height_scanner"]
        starts, directions = tensor(sensor.ray_starts), tensor(sensor.ray_directions)
        if starts.ndim == 3:
            starts, directions = starts[0], directions[0]
        if len(starts) != 187:
            raise ValueError("Thunder 309-value critic requires its 187-ray height scan")
        self.height_perm = reflected_ray_indices(starts, directions)

    @staticmethod
    def reverse_joints(value):
        if value.shape[-1] != 16:
            raise ValueError("Thunder requires 16 actions/joint vectors")
        # Axial velocity/torque reflection and opposite paired motor conventions
        # require a minus sign for hip, thigh, calf AND wheel joints.
        return -value[...,list(JOINT_PERM)]

    @staticmethod
    def permute_joints(value):
        if value.shape[-1] != 16:
            raise ValueError("Thunder requires 16 action standard deviations")
        return value[...,list(JOINT_PERM)]

    def reverse_frame(self, value):
        if value.shape[-1] != 53:
            raise ValueError("Thunder actor frame must contain 53 values")
        return torch.cat((value[...,:3]*value.new_tensor([-1.,1.,-1.]),
            value[...,3:6]*value.new_tensor([1.,-1.,1.]),
            value[...,6:9]*value.new_tensor([1.,-1.,-1.]),
            -value[...,9:21][...,list(JOINT_PERM[:12])],
            self.reverse_joints(value[...,21:37]),self.reverse_joints(value[...,37:53])),dim=-1)

    def reverse_critic(self, value):
        if value.shape[-1] != 309:
            raise ValueError("Thunder critic must contain 309 values")
        index = self.height_perm.to(value.device)
        return torch.cat((value[...,:3]*value.new_tensor([1.,-1.,1.]),
            value[...,3:6]*value.new_tensor([-1.,1.,-1.]),
            value[...,6:9]*value.new_tensor([1.,-1.,1.]),
            value[...,9:12]*value.new_tensor([1.,-1.,-1.]),
            -value[...,12:24][...,list(JOINT_PERM[:12])],
            self.reverse_joints(value[...,24:40]),self.reverse_joints(value[...,40:56]),
            value[...,56:243].index_select(-1,index),
            value[...,243:255][...,list(GAIN_PERM[:12])],
            value[...,255:271][...,list(GAIN_PERM)],value[...,271:273],
            self.reverse_joints(value[...,273:289]),self.reverse_joints(value[...,289:305]),
            value[...,305:309][...,list(FOOT_PERM)]),dim=-1)

    def reverse_obs(self, obs):
        result = obs.clone()
        for key,value in obs.items():
            if key == "policy":
                if value.shape[-1] != 530:
                    raise ValueError("Thunder policy history must be 10 frame-major frames")
                result[key] = self.reverse_frame(value.reshape(*value.shape[:-1],10,53)).reshape_as(value)
            elif key == "single_obs":
                result[key] = self.reverse_frame(value)
            elif key == "critic":
                result[key] = self.reverse_critic(value)
            else:
                raise ValueError("Unknown Thunder observation group: " + key)
        return result

    def data_augmentation(self, obs, actions):
        return torch.cat([obs,self.reverse_obs(obs)],dim=0),torch.cat([actions,self.reverse_joints(actions)],dim=0)


class ThunderSymmetry(Go2Symmetry):
    """Reuse only upstream batch ordering/duplication, never its 12-action mapping."""
    def __init__(self, env):
        self.mapper = ThunderSymmetryMapper(env)
        self.num_aug = 2
