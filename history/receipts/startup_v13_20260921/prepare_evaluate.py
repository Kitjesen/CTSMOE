from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'full_episode_v12_20260921'
for name in ('analyze.py', 'gates.py', 'plot_comparison.py', 'flat25_geometry.py', 'test_gates.py'):
    target = HERE/name
    if target.exists():
        raise FileExistsError(target)
    target.write_bytes((SOURCE/name).read_bytes())
text = (SOURCE/'evaluate.py').read_text(encoding='utf-8')
text = text.replace("p.add_argument('--smoke'", "p.add_argument('--ramp-starts',action='store_true',help='Separate 0.2-second ramp diagnostic; direct gates remain separate')\np.add_argument('--smoke'")
text = text.replace("'omni_full_episode_v1')", "'omni_full_episode_v1', 'omni_full_episode_v13')")
text = text.replace("for mode in ('direct','settle')", "for mode in (('ramp','settle') if a.ramp_starts else ('direct','settle'))")
text = text.replace('guard=ActuatorGuard(raw)', 'guard=ActuatorGuard(raw)\n    from reset_targets import install_reset_target_sync\n    reset_receipt=install_reset_target_sync(raw)')
text = text.replace("velocities=torch.tensor([c['velocity'] for c in cases],device=raw.device)",
                    "velocities=torch.tensor([c['velocity'] for c in cases],device=raw.device)\n    ramp_mask=torch.tensor([c['mode']=='ramp' for c in cases],device=raw.device)")
text = text.replace("command.commands[:]=velocities*(elapsed>=settle)[:,None]",
                    "factor=torch.where(ramp_mask, (elapsed/.2).clamp(0.,1.), (elapsed>=settle).float())\n        command.commands[:]=velocities*factor[:,None]")
text = text.replace("expected_initial_commands = velocities*(settle == 0.)[:,None]",
                    "expected_initial_commands = velocities*((settle == 0.) & ~ramp_mask)[:,None]\n    torch.testing.assert_close(robot.data.joint_pos_target, robot.data.joint_pos, atol=1e-6, rtol=0.)")
text = text.replace('initial_episode_age_zero_verified=True,',
                    'initial_episode_age_zero_verified=True, reset_targets=reset_receipt,\n                  ramp_diagnostic=a.ramp_starts, direct_gate_protocol=not a.ramp_starts,')
with (HERE/'evaluate.py').open('x', encoding='utf-8') as stream:
    stream.write(text)
