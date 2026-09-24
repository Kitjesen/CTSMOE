"""Only extend measured traces; retain the corrected matched evaluation dynamics."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'early_ablation_v15_20260921/evaluation_audit_20260922/evaluate_corrected.py'


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(old)
    return text.replace(old, new, 1)


text = SOURCE.read_text(encoding='utf-8')
text = once(text, 'from early_protocol import (',
    "sys.path.insert(0, '/root/autodl-tmp/cts-v15-evaluation-audit-20260922')\n\nfrom early_protocol import (")
text = once(text, '"platform_stop", "pit_all_supported",',
    '"wheel_ground_force_z", "flat_wheel_clearance", "platform_stop", "pit_all_supported",')
text = once(text, '                wheel_ground_force=ground_force,',
    '                wheel_ground_force=ground_force,\n'
    '                wheel_ground_force_z=torch.stack([sensor.data.force_matrix_w[:, 0, 0, 2] for sensor in pit.ground_contacts], dim=1),\n'
    '                flat_wheel_clearance=data.body_link_pos_w[:, pit.wheel_ids, 2] - origins[:, 2:3] - .093,')
(HERE/'evaluate_measured.py').write_text(text, encoding='utf-8')
path = HERE/'evaluate_pair.py'
text = path.read_text(encoding='utf-8')
text = once(text, "str(AUDIT/'evaluate_corrected.py')", "str(Path(__file__).with_name('evaluate_measured.py'))")
path.write_text(text, encoding='utf-8')
