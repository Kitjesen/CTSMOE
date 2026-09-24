"""Create auditable V19 qualification drivers from the validated V18 interface."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'phase_pose_v18_20260922'


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(old)
    return text.replace(old, new, 1)


def main():
    for name in ('preflight.py', 'run_preflight.py', 'verify_preflight.py', 'evaluate_pair.py'):
        text = (OLD/name).read_text(encoding='utf-8')
        if name == 'preflight.py':
            text = once(text, '    from phase_pose import snapshot as pose_snapshot\n',
                '    from phase_pose import snapshot as pose_snapshot\n'
                '    from omni_lift import snapshot as lift_snapshot\n')
            text = once(text, '        result["phase_pose"] = pose_snapshot(env)\n',
                '        result["phase_pose"] = pose_snapshot(env)\n'
                '        result["omni"] = env.command_manager.get_term("base_velocity").omni_snapshot()\n'
                '        result["lift"] = lift_snapshot(env)\n'
                '        result["allocation"] = env.terrain_allocation.snapshot()\n')
            text = once(text, '    assert phase_coverage["flat"] == [1., 1.]\n',
                '    assert phase_coverage["flat"] == [1., 1.]\n'
                '    assert all(min(row) > 0 for row in receipt["omni"]["kind_counts_by_learner"])\n'
                '    assert all(min(row) > 0 for row in receipt["omni"]["stop_restart_by_learner"])\n'
                '    assert receipt["allocation"]["pending_moves"] == 0\n'
                '    for name, weight in receipt["allocation"]["target_weights"].items():\n'
                '        assert abs(receipt["allocation"]["actual_counts"][name] - 4096*weight) <= 3\n')
        elif name == 'verify_preflight.py':
            text = once(text, "    s = final['settings']\n", "    s = final['settings']\n"
                "    assert s['omni']['profile'] == 'omni_full_episode_v19'\n"
                "    assert s['allocation']['bounds_profile'] == 'omni_repair_v19'\n"
                "    assert s['allocation']['target_weights']['flat'] == .40\n"
                "    assert all(min(row) > 0 for row in s['omni']['kind_counts_by_learner'])\n")
        elif name == 'evaluate_pair.py':
            text = once(text, "        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path,expected in hashes.items())",
                "        for suite in ('terrain83', 'flat25'):\n"
                "            subprocess.run([sys.executable, str(Path(__file__).with_name('omni_metrics.py')), str(out/suite)], check=True)\n"
                "        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path,expected in hashes.items())")
        with (HERE/name).open('x', encoding='utf-8') as output:
            output.write(text)


if __name__ == '__main__':
    main()
