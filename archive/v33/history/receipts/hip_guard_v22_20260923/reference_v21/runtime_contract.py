"""CPU-only identity, file audit, and V21 coverage qualification contracts."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/root/autodl-tmp/cts-omni-repair-v19-20260922')
BASE = Path('/root/autodl-tmp/workspace/doso-thunder-cts-official-20260917')
COMPLETED = 33865
COMMON_STEP = 812760
STEPS = 3000
NUM_ENVS = 4096
SOURCE_FILES = 86
REVIEWED_COVERAGE_SHA256 = '54788c3c2179de8f9afb62fd0915b3c35ffa7a40691d754cb9a4187ecefa442a'
IDENTITY_NAME = 'SOURCE_IDENTITY.json'
REVIEWED_IDENTITY_SHA256 = '2666c590ac843ac2626a18286490fb44557cebdba6bfb1d3aaf7b639a373de62'
SOURCE_RELATIVE = Path('speed_brake_v20/training_33174_to_34823/model_33864.pt')
PROFILE = 'stair_terminal_columns_v21'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_identity(path):
    if file_sha256(path) != REVIEWED_IDENTITY_SHA256:
        raise ValueError('Source identity receipt differs from the reviewed original bytes')
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (value.get('completed_updates') != COMPLETED or value.get('common_step_counter') != COMMON_STEP
            or not re.fullmatch('[0-9a-f]{64}', value.get('sha256', ''))
            or not isinstance(value.get('bytes'), int) or value['bytes'] <= 0
            or not str(value.get('path', '')).replace('\\', '/').endswith('/' + SOURCE_RELATIVE.as_posix())):
        raise ValueError('Invalid preserved 33865 identity receipt')
    return value


def verify_source(identity):
    source = Path(identity['path'])
    if source.stat().st_size != identity['bytes'] or file_sha256(source) != identity['sha256']:
        raise ValueError('Preserved source checkpoint identity changed')
    return source


def verify_files(folder, files, count=None):
    folder = Path(folder)
    if not isinstance(files, dict) or not files or (count is not None and len(files) != count):
        raise ValueError('Unexpected audited file count')
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.py':
            raise ValueError(f'Unsafe runtime manifest member: {name}')
        if not re.fullmatch('[0-9a-f]{64}', expected) or file_sha256(folder / relative) != expected:
            raise ValueError(f'Frozen runtime changed: {folder / relative}')
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob('*.py')}
    if actual != set(files):
        raise ValueError(f'Untracked/missing Python runtime files: {sorted(actual ^ set(files))}')


def verify_prepared(here):
    here = Path(here)
    prepared = json.loads((here / 'PREPARED.json').read_text(encoding='utf-8'))
    if (prepared.get('source_file_count') != SOURCE_FILES
            or len(prepared.get('source_files', {})) != SOURCE_FILES
            or prepared.get('changed_source_files') != ['train.py']
            or prepared.get('added_runtime_files') != ['stair_transition_coverage.py']):
        raise ValueError('Unexpected V21 source change scope')
    files = prepared['files']
    source_files = prepared['source_files']
    if (set(files) != set(source_files) | {'stair_transition_coverage.py'}
            or sorted(name for name in source_files if files[name] != source_files[name]) != ['train.py']):
        raise ValueError('V21 manifest exceeds train insertion and coverage module')
    if files['stair_transition_coverage.py'] != REVIEWED_COVERAGE_SHA256:
        raise ValueError('Prepared coverage is not the reviewed module')
    verify_files(here / 'code', files, SOURCE_FILES + 1)
    # Existing V20 runtime remains part of the qualification, before and after execution.
    verify_files(Path(prepared['source_code']), source_files, SOURCE_FILES)
    payload = json.loads((here / 'RUNTIME_PREPARE_MANIFEST.json').read_text(encoding='utf-8'))['files']
    for name, expected in payload.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_sha256(here / relative) != expected:
            raise ValueError(f'V21 preparation payload changed: {name}')
    identity = load_identity(here / IDENTITY_NAME)
    if prepared.get('source_identity') != identity:
        raise ValueError('Prepared source identity disagrees with payload')
    verify_source(identity)
    return identity, prepared


def validate_coverage(receipt):
    if (receipt.get('profile') != PROFILE or receipt.get('generated_cell_count') != 880
            or receipt.get('selected_cell_count') != 44):
        raise ValueError('Incomplete V21 generated mesh coverage')
    config = receipt['config']
    if (config.get('columns_by_kind') != {'stairs_up': [8], 'stairs_down': [11]}
            or config.get('generated_rows') != list(range(22))
            or config.get('active_original_rows') != list(range(10))
            or config.get('geometry_only') is not True
            or config.get('rewards_commands_terminations_changed') is not False):
        raise ValueError('Coverage whitelist or geometry-only scope changed')
    meshes = receipt['mesh_receipts']
    if len(meshes) != 880 or {(r['row'], r['column']) for r in meshes} != {
            (row, col) for row in range(22) for col in range(40)}:
        raise ValueError('Missing or repeated generated terrain cells')
    for mesh in meshes:
        selected = mesh['column'] in (8, 11)
        if mesh.get('selected') is not selected or mesh.get('origin_unchanged') is not True:
            raise ValueError('Cell whitelist or origin changed')
        if mesh['origin_before'] != mesh['origin_after']:
            raise ValueError('Terrain origin changed')
        if selected:
            if (mesh['kind'] != ('stairs_up' if mesh['column'] == 8 else 'stairs_down')
                    or mesh['before_sha256'] == mesh['after_sha256']
                    or not all(mesh.get(k) is True for k in (
                        'inner_vertices_unchanged', 'inner_nondegenerate_triangles_unchanged', 'xy_unchanged'))):
                raise ValueError('Selected mesh lacks geometry preservation proof')
            for stem in ('inner_vertices', 'inner_triangle_vertex_positions', 'inner_nondegenerate_triangles'):
                if mesh[stem + '_before_sha256'] != mesh[stem + '_after_sha256']:
                    raise ValueError('Protected inner geometry changed')
            if (mesh.get('cap_original_level_count') != 1
                    or mesh.get('changed_faces_outside_inner_square') is not True
                    or mesh.get('changed_faces_inner_square_overlap_count') != 0
                    or mesh.get('changed_face_count', 0) <= 0
                    or mesh.get('changed_faces_min_axis_separation_m') is None
                    or mesh['changed_faces_min_axis_separation_m'] < 0):
                raise ValueError('Missing original-height or whole-triangle inner-region protection proof')
        elif mesh['before_sha256'] != mesh['after_sha256']:
            raise ValueError('Nonselected terrain changed')
    counts = receipt['assignments']
    learners = counts['by_learner']
    if counts['total'] != NUM_ENVS or sum(g['total'] for g in learners.values()) != NUM_ENVS:
        raise ValueError('Missing actual environment assignments')
    if set(learners) != {'teacher', 'student'}:
        raise ValueError('Missing learner groups')
    for label, expected in (('teacher', 3072), ('student', 1024)):
        group = learners[label]
        if group['total'] != expected or sum(v['total'] for v in group['by_type'].values()) != expected:
            raise ValueError('Learner partition changed')
        if set(group['by_type']) != set(config['expected_column_names']):
            raise ValueError('Missing terrain type assignments')
        for kind, values in group['by_type'].items():
            if values['total'] != values['coverage'] + values['original']:
                raise ValueError('Inconsistent type assignment counts')
            levels = values['coverage_by_level']
            if len(levels) != 22 or sum(levels) != values['coverage'] or any(v < 0 for v in levels):
                raise ValueError('Invalid coverage-by-level counts')
            if kind in ('stairs_up', 'stairs_down'):
                if min(values['coverage'], values['original']) <= 0 or any(levels[10:]):
                    raise ValueError('Both coverage and original stair cohorts must exist per learner')
            elif values['coverage'] != 0:
                raise ValueError('Non-stair terrain received coverage')
        if group['coverage'] != sum(v['coverage'] for v in group['by_type'].values()):
            raise ValueError('Inconsistent learner coverage count')
    if (counts['coverage_total'] != sum(g['coverage'] for g in learners.values())
            or abs(counts['fraction'] - counts['coverage_total'] / NUM_ENVS) > 1e-12):
        raise ValueError('Inconsistent total coverage count')
