"""Copy the qualified runtime checks with explicit V18 phase verification."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'posture_stop_v17_20260922'
text = (OLD / 'preflight.py').read_text()
text = text.replace('    from posture_stop import snapshot\n',
    '    from posture_stop import snapshot as base_snapshot\n'
    '    from phase_pose import snapshot as pose_snapshot\n'
    '    def snapshot(env):\n'
    '        result = base_snapshot(env)\n'
    '        result["phase_pose"] = pose_snapshot(env)\n'
    '        for name in ("early_actual_hip_l1", "early_actual_thigh_calf_l1"):\n'
    '            assert env.reward_manager.get_term_cfg(name).weight == -.20\n'
    '        assert result["phase_pose"]["by_terrain"]["flat"]["gate_min"] == 1.\n'
    '        return result\n')
text = text.replace('    previous_stop = torch.zeros',
    '    phase_coverage = {name: [1., 0.] for name in ("flat", "stairs", "pit")}\n'
    '    previous_stop = torch.zeros')
text = text.replace('                receipt = snapshot(raw)',
    '                receipt = snapshot(raw)\n'
    '                for name, value in receipt["phase_pose"]["by_terrain"].items():\n'
    '                    phase_coverage[name][0] = min(phase_coverage[name][0], value["gate_min"])\n'
    '                    phase_coverage[name][1] = max(phase_coverage[name][1], value["gate_max"])')
text = text.replace("    (out/'FINALIZED.json').write_text", 
    '    assert phase_coverage["flat"] == [1., 1.]\n'
    '    assert phase_coverage["pit"][0] < .1 and phase_coverage["pit"][1] > .9\n'
    '    assert phase_coverage["stairs"][0] < .1 and phase_coverage["stairs"][1] > .9\n'
    "    (out/'FINALIZED.json').write_text")
text = text.replace("        settings=receipt, qualification=", "        settings=receipt, phase_coverage=phase_coverage, qualification=")
(HERE / 'preflight.py').write_text(text)
for name in ('run_preflight.py', 'verify_preflight.py', 'launch_training.py'):
    text = (OLD / name).read_text()
    text = text.replace('V17', 'V18')
    if name == 'verify_preflight.py':
        text = text.replace("    s = final['settings']", 
            "    assert final['phase_coverage']['flat'] == [1., 1.]\n"
            "    assert final['settings']['phase_pose']['profile'] == 'phase_pose_v18'\n"
            "    s = final['settings']")
    (HERE / name).write_text(text)
print('Drivers prepared')
