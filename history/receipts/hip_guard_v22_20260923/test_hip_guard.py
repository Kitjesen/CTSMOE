import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

CODE = Path(__file__).parent / 'candidate_attempt03/code'
sys.path.insert(0, str(CODE))
from hip_contract import INWARD_SIGNS, joint_contract  # noqa: E402


@pytest.fixture
def action(monkeypatch):
    module = ModuleType('isaaclab.envs.mdp.actions')

    class JointPositionAction:
        def __init__(self, cfg, env):
            self._asset = env.scene['robot']
            self._joint_ids = cfg.joint_ids
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
            self._scale = .25
            self._raw_actions = torch.zeros_like(self._offset)
            self._processed_actions = torch.zeros_like(self._offset)

        def process_actions(self, actions):
            self._raw_actions.copy_(actions)
            self._processed_actions = actions * self._scale + self._offset

        def reset(self, env_ids=None):
            self._raw_actions[:] = 0

    module.JointPositionAction = JointPositionAction
    monkeypatch.setitem(sys.modules, 'isaaclab.envs.mdp.actions', module)
    for name in ('guard_actions', 'hip_guard'):
        sys.modules.pop(name, None)
    hip = importlib.import_module('hip_guard')
    names = [f'{side}_{part}_joint' for part in ('hip', 'thigh', 'calf', 'foot')
             for side in ('fl', 'fr', 'rl', 'rr')]
    ids = [names.index(f'{side}_{part}_joint') for side in ('fr', 'fl', 'rr', 'rl')
           for part in ('hip', 'thigh', 'calf')]
    defaults = torch.zeros(8, 16)
    for name in INWARD_SIGNS:
        defaults[:, names.index(name)] = joint_contract(name)['default_rad']
    hard = torch.tensor([-.41, .41]).repeat(8, 16, 1)
    data = SimpleNamespace(default_joint_pos=defaults, joint_pos=defaults.clone(), soft_joint_pos_limits=hard.clone()*.9)
    robot = SimpleNamespace(joint_names=names, data=data,
                            root_physx_view=SimpleNamespace(get_dof_limits=lambda: hard))
    env = SimpleNamespace(scene={'robot': robot}, step_dt=.02,
                           cfg=SimpleNamespace(actuator_profile='rs0402_transient_v4'))
    obj = hip.HipGuardedJointPositionAction(SimpleNamespace(joint_ids=ids), env)
    return obj, robot, hard


def test_inward_requests_are_capped_but_outward_room_is_retained(action):
    obj, robot, hard = action
    before_state = robot.data.joint_pos.clone()
    signs = torch.tensor([INWARD_SIGNS[f'{side}_hip_joint'] for side in ('fr','fl','rr','rl')])
    hip_ids = [0, 3, 6, 9]
    inward = torch.zeros(8, 12)
    inward[:, hip_ids] = signs * 20
    obj.process_actions(inward)
    torch.testing.assert_close(obj._processed_actions[:, hip_ids] * signs, torch.full((8,4), .12))
    outward = -inward
    for _ in range(3):
        previous = obj._processed_actions.clone()
        obj.process_actions(outward)
        assert (obj._processed_actions-previous).abs().max() <= .240001
    torch.testing.assert_close(obj._processed_actions[:, hip_ids] * signs, torch.full((8,4), -.369))
    torch.testing.assert_close(robot.data.joint_pos, before_state, rtol=0, atol=0)
    torch.testing.assert_close(hard, obj.hip_guard_physical_limits_before, rtol=0, atol=0)


def test_new_bounds_do_not_modify_thigh_calf_or_wheel(action):
    _, robot, hard = action
    other = [i for i,n in enumerate(robot.joint_names) if n not in INWARD_SIGNS]
    torch.testing.assert_close(robot.data.soft_joint_pos_limits[:,other], hard[:,other]*.9, rtol=0, atol=0)


def test_subset_reset_does_not_teleport_actual_joints(action):
    obj, robot, _ = action
    ids = torch.tensor([2,5])
    robot.data.joint_pos[ids] = .3
    original = robot.data.joint_pos.clone()
    previous = obj._previous.clone()
    obj.reset(ids)
    torch.testing.assert_close(robot.data.joint_pos, original, rtol=0, atol=0)
    torch.testing.assert_close(obj._previous[[0,1,3,4,6,7]], previous[[0,1,3,4,6,7]])
    assert bool((obj._previous[ids] >= obj._lower[ids]).all())
    assert bool((obj._previous[ids] <= obj._upper[ids]).all())


def test_reset_motor_zero_randomization_is_verified_not_removed(action):
    from hip_guard import validate_offset
    nominal = torch.full((8,), .12)
    validate_offset(nominal, nominal, None, (-.035, .035))
    validate_offset(nominal, nominal + torch.linspace(-.035, .035, 8), nominal, (-.035, .035))
    with pytest.raises(ValueError):
        validate_offset(nominal, nominal + .04, nominal, (-.035, .035))
    with pytest.raises(AssertionError):
        validate_offset(nominal, nominal, nominal + .08, (-.035, .035))


def test_actual_checkpoint_save_contract_round_trips():
    import ast
    from hip_contract import PROFILE, contract
    from policy_contract import checkpoint_profile, export_contract
    tree = ast.parse((CODE/'train.py').read_text(encoding='utf-8'))
    saves = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'update'
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'saved'
             and any(k.arg == 'actuator_contract' for k in node.keywords)]
    assert len(saves) == 1
    values = {}
    namespace = dict(export_contract=export_contract, actuator_profile='rs0402_transient_v4',
                     HIP_PROFILE=PROFILE, hip_contract=contract)
    for field in saves[0].keywords:
        if field.arg in ('actuator_profile','actuator_contract','hip_guard_profile','hip_contract'):
            values[field.arg] = eval(compile(ast.Expression(field.value),'<save-contract>','eval'), namespace)
    assert checkpoint_profile(values) == 'rs0402_transient_v4'
    assert values['hip_guard_profile'] == PROFILE
    assert values['hip_contract'] == contract()
    assert 'hip_contract.json' in (CODE/'train.py').read_text(encoding='utf-8')


def test_urdf_kinematics_independently_confirms_inward_signs():
    path = Path(__file__).parents[2] / 'recordings/student_33865_20260922/trace/runtime.urdf'
    root = ET.parse(path).getroot()
    joints = list(root.findall('joint'))

    def wheels(inward):
        poses = {'base_link': np.eye(4)}
        pending = joints.copy()
        while pending:
            count = len(pending)
            for joint in pending.copy():
                parent = joint.find('parent').get('link')
                if parent not in poses:
                    continue
                name = joint.get('name')
                origin = joint.find('origin')
                transform = np.eye(4)
                if origin is not None:
                    transform[:3,3] = np.fromstring(origin.get('xyz','0 0 0'), sep=' ')
                    transform[:3,:3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy','0 0 0'), sep=' ')).as_matrix()
                q = 0.
                if name in INWARD_SIGNS:
                    q = INWARD_SIGNS[name] * inward
                elif '_thigh_joint' in name:
                    q = -.93 if name.startswith(('fr_', 'rl_')) else .93
                elif '_calf_joint' in name:
                    q = 1.96 if name.startswith(('fr_', 'rl_')) else -1.96
                if joint.get('type') != 'fixed':
                    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                    rot = np.eye(4)
                    rot[:3,:3] = Rotation.from_rotvec(axis*q).as_matrix()
                    transform = transform @ rot
                poses[joint.find('child').get('link')] = poses[parent] @ transform
                pending.remove(joint)
            assert len(pending) < count, 'Disconnected URDF tree'
        return {side: poses[f'{side}_foot_Link'][:3,3] for side in ('fr','fl','rr','rl')}

    old, candidate, excessive = wheels(.20), wheels(.12), wheels(.369)
    for side in old:
        assert abs(candidate[side][1]) > abs(old[side][1]) > abs(excessive[side][1])
        assert abs(candidate[side][1]) == pytest.approx(.23735, abs=.0001)
