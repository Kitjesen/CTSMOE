"""Derive the anchored update from the pinned upstream CTS implementation."""
import ast
import hashlib
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
with tarfile.open(HERE.parent.parent/'source.tgz') as archive:
    member = next(m for m in archive.getmembers() if m.name.endswith('rsl_rl/algorithms/moe_cts.py'))
    base = archive.extractfile(member).read()
assert hashlib.sha256(base).hexdigest() == '136a1c1082243ae1c6654307ffcf32de2a99cf6c4d6581419ada59f288a058f6'
source = base.decode()
tree = ast.parse(source)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoECTS')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'update')
update = '\n'.join(source.splitlines()[method.lineno-1:method.end_lineno]) + '\n'


def replace(old, new):
    global update
    assert update.count(old) == 1, old
    update = update.replace(old, new)


replace('        mean_value_loss = 0', '        self.anchor_metrics = dict(teacher=0., student=0.)\n        mean_anchor_main = 0.0\n        mean_anchor_student = 0.0\n        mean_value_loss = 0')
replace('            # RND loss\n            # TODO: Move this processing to inside RND module.', '''            anchor_loss = self.main_anchor_loss(obs_batch, teacher_samples)
            if self.anchor_coefficients['main'] is None and obs_batch['anchor_mask'].any():
                candidate = gradient_ratio_coefficient(
                    loss, anchor_loss, tuple(self.policy.actor.parameters()) +
                    tuple(self.policy.teacher_encoder.parameters()), ratio=.1,
                    statistics=self.anchor_calibration['main'])
                if candidate > 0:
                    self.anchor_coefficients['main'] = candidate
            loss = loss + (self.anchor_coefficients['main'] or 0.) * anchor_loss
            mean_anchor_main += anchor_loss.item()

            # RND loss''')
replace('            self.optimizer_stu_enc.zero_grad()', '''            student_anchor = self.student_anchor_loss(obs_batch[teacher_samples:])
            if self.anchor_coefficients['student'] is None and obs_batch['anchor_mask'][teacher_samples:].any():
                candidate = gradient_ratio_coefficient(
                    student_loss, student_anchor, tuple(self.policy.student_moe_encoder.parameters()), ratio=.1,
                    statistics=self.anchor_calibration['student'])
                if candidate > 0:
                    self.anchor_coefficients['student'] = candidate
            student_loss = student_loss + (self.anchor_coefficients['student'] or 0.) * student_anchor
            mean_anchor_student += student_anchor.item()
            self.optimizer_stu_enc.zero_grad()''')
replace('        return loss_dict', '''        loss_dict.update(anchor_main=mean_anchor_main/num_updates,
                         anchor_main_teacher=self.anchor_metrics['teacher']/num_updates,
                         anchor_main_student=self.anchor_metrics['student']/num_updates,
                         anchor_student=mean_anchor_student/num_updates,
                         anchor_main_coefficient=self.anchor_coefficients['main'] or 0.,
                         anchor_student_coefficient=self.anchor_coefficients['student'] or 0.)
        return loss_dict''')
# Preserve upstream executable statements; remove inherited comment-only TODOs.
update = '\n'.join(line for line in update.splitlines() if not line.lstrip().startswith('#')) + '\n'
header = (HERE / 'algorithm_header.txt').read_text(encoding='utf-8')
destination = HERE.parent.parent / 'anchor_cts.py'
destination.write_text(header + '\n' + update, encoding='utf-8')
ast.parse(destination.read_text(encoding='utf-8'))
