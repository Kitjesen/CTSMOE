"""Fixed test matrix; this module changes no training configuration."""
import itertools
import json


def load_student_selection(path, protocol):
    """This deployment check has one student, unlike the paired training gate.

    Keep source membership, exact schema and iteration-contract validation;
    requiring an unused teacher would misrepresent this ONNX-only experiment.
    """
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len(rows) != 1 or len(protocol['sources']) != 1:
        raise ValueError('Require exactly one deployment candidate and source')
    row = rows[0]
    if not isinstance(row, dict) or set(row) != {'path', 'label', 'teacher'}:
        raise ValueError('Invalid deployment selection schema')
    if row['teacher'] is not False or row['path'] != protocol['sources'][0]['path']:
        raise ValueError('Student must match the contracted source')
    if not isinstance(row['label'], str) or not row['label']:
        raise ValueError('Missing candidate label')
    return rows


def build_release_cases(selection, suite, original_builder):
    if suite != 'stairend' or len(selection) != 1 or selection[0]['teacher']:
        raise ValueError('Student-only continuous stairs protocol required')
    templates = original_builder(selection, 'terrain83')
    rows = []
    for terrain, level, speed, mode, landing in itertools.product(
            ('stairs_up', 'stairs_down'), (3, 6, 9), (.3, .6, 1.2),
            ('direct', 'settle'), ('continue', 'stop')):
        template = next(c for c in templates if c['terrain'] == terrain)
        rows.append(dict(template, id=len(rows), level=level, mode=mode,
                         velocity=[speed, 0., 0.], landing_mode=landing))
    assert len(rows) == 72
    return rows
