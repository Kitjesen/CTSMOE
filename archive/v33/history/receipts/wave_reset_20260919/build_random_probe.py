"""One-second matched reset tests with the actual training pose and velocity ranges."""
from pathlib import Path

folder = Path(__file__).resolve().parent
source = (folder/'probe_motion.py').read_text()
for line in (
    '    cfg.events.reset_base.params["pose_range"] = {k: (0., 0.) for k in ("x", "y", "z", "roll", "pitch", "yaw")}\n',
    '    cfg.events.reset_base.params["velocity_range"] = dict(cfg.events.reset_base.params["pose_range"])\n',
    '    cfg.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))\n',
    '    cfg.events.reset_base.params["pose_range"]["z"] = (args.reset_z, args.reset_z)\n',
    '        torch.testing.assert_close(initial[:, :2], raw.scene.env_origins[:, :2], rtol=0, atol=1e-5)\n',
):
    assert source.count(line) == 1, line
    source = source.replace(line, '')
source = source.replace('range(raw.max_episode_length):', 'range(50):')
source = source.replace('scope="Forward 4 m or first termination, maximum 25 s; no learning; not a full terrain success certification"',
                        'scope="First one second only, training pose/joint/velocity reset ranges; not a traversal test"')
source = source.replace('three nominal-center repeats', 'three random-pose/velocity draws')
(folder/'probe_random_start.py').write_text(source)
