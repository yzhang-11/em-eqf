"""Plan, preserve, relocate, and verify revision artifacts in the existing layout."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
from datetime import datetime, timezone

ROOT = Path('/root/autodl-tmp/em_eqf')
DEST = ROOT / 'experiments/revision'
RECORD = DEST / 'reports/layout'
ARCHIVE = 'experiments/revision/archive/development'
GROUPS = {'review_v4': 'main', 'review_v4_controls': 'controls',
          'review_v4_generation_baselines': 'generation_baselines'}
SCRIPTS = {
    'run_revision_v4_initial.py': 'run/run_revision_initial.py',
    'run_supervised_v4.py': 'run/run_revision_supervised.py',
    'run_reviewer_controls.py': 'run/run_revision_controls.py',
    'run_generation_followups.py': 'run/run_revision_generation.py',
    'run_review_baselines.py': 'run/run_revision_references.py',
    'prepare_reviewer_controls.py': 'data/prepare_revision_controls.py',
    'prepare_generation_followups.py': 'data/prepare_revision_generation.py',
    'analyze_primary_v4.py': 'analyze/analyze_revision_supervised.py',
    'analyze_reviewer_controls.py': 'analyze/analyze_revision_controls.py',
    'classification_block_sensitivity.py': 'analyze/classification_block_sensitivity.py',
    'compare_partial_scratch_controls.py': 'analyze/compare_partial_scratch_controls.py',
    'compare_generation_magnitude.py': 'analyze/compare_generation_magnitude.py',
    'compare_generation_time.py': 'analyze/compare_generation_time.py',
    'evaluate_dynamic_b_reference.py': 'analyze/evaluate_dynamic_b_reference.py',
    'evaluate_full_window_calibration.py': 'analyze/evaluate_full_window_calibration.py',
    'evaluate_full_magnitude_test.py': 'analyze/evaluate_full_magnitude_test.py',
    'summarize_full_calibration.py': 'summarize/summarize_full_calibration.py',
    'export_generation_data_protocol.py': 'summarize/export_generation_data_protocol.py',
    'audit_window_protocol_v4.py': 'diagnostics/audit_revision_windows.py',
    'check_cached_density_replay.py': 'diagnostics/check_cached_density_replay.py',
    'check_inference_capacity.py': 'diagnostics/check_inference_capacity.py',
    'check_gpu_sampling.py': 'diagnostics/check_gpu_sampling.py',
    'validate_attention_cache_fix.py': 'diagnostics/validate_attention_cache_fix.py',
    'verify_legacy_sampling_readonly.py': 'diagnostics/verify_legacy_sampling_readonly.py',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def make_maps():
    maps = {'revision_audit': ARCHIVE, 'data/DATA_LAYOUT.md': 'docs/guides/revision_data_layout.md'}
    for old, new in GROUPS.items():
        base, target = 'experiments/' + old, 'experiments/revision/' + new
        maps[base] = target
        maps[base + '/analysis'] = target + '/reports/supervised' if new == 'main' else target + '/reports'
        for p in (ROOT / base).iterdir():
            if p.suffix in {'.pid', '.log'}:
                maps[str(p.relative_to(ROOT))] = target + '/logs/' + p.name
    main = 'experiments/review_v4/'
    for name in ['calibration', 'generation_comparison', 'generation_comparison_sensitivity',
                 'magnitude_test', 'window_protocol_audit']:
        maps[main + name] = 'experiments/revision/main/reports/' + name
    maps[main + 'baselines'] = 'experiments/revision/main/runs/references'
    for p in (ROOT / 'experiments/review_v4').glob('calibration_*'):
        if p.is_dir():
            maps[str(p.relative_to(ROOT))] = 'experiments/revision/archive/calibration/' + p.name
    base = 'experiments/review_v4_generation_baselines/'
    for name in ['time_comparison', 'time_nll_summary.csv', 'generation_periods.csv', 'swc_cross_pipeline_and_tail_audit.json']:
        maps[base + name] = 'experiments/revision/generation_baselines/reports/' + name
    maps[base + 'smoke'] = 'experiments/revision/archive/smoke/generation_baselines'
    maps[base + 'preparation_archive'] = 'experiments/revision/archive/preparation/generation_baselines'
    maps[main + 'magnitude_test/preparation_archive'] = 'experiments/revision/archive/preparation/magnitude_test'
    for name in ['preflight', 'large_target_support.csv', 'transfer_loading_audit.json']:
        maps['experiments/review_v4_controls/' + name] = 'experiments/revision/controls/reports/' + name
    for name in ['review_v2', 'review_v2_smoke', 'review_v3', 'review_v4_smoke', 'review_v4_controls_smoke']:
        maps['experiments/' + name] = 'experiments/revision/archive/' + name
    maps['revision_audit/mc_20261102'] = 'experiments/revision/main/reports/completeness'
    maps['revision_audit/mc_20261102/raw_scedc_snapshot_20261102'] = 'data/SCEDC/raw/scsn_annual_snapshot_20261102'
    maps['revision_audit/mc_20261102/scsn_untruncated_current.csv'] = 'data/SCEDC/processed/scsn_all_types_snapshot_20261102.csv'
    for name, target in SCRIPTS.items():
        maps['revision_audit/' + name] = 'scripts/' + target
    for task in ['classifier', 'regressor', 'lstm_regressor']:
        maps['data/ChuanDian/review_v2/' + task] = 'data/ChuanDian/' + task
    old = json.loads((ROOT / 'revision_audit/data_layout_20261104/plan.json').read_text())
    for row in old['aliases']:
        maps[str(Path(row['old']).relative_to(ROOT))] = str(Path(row['target']).relative_to(ROOT))
    return dict(sorted(maps.items(), key=lambda x: len(x[0]), reverse=True))


def relocate(relative, maps):
    for old, new in maps.items():
        if relative == old or relative.startswith(old + '/'):
            return new + relative[len(old):]
    return relative


def replace_paths(text, maps):
    # One pass: a replacement is never fed into a second mapping.
    pattern = re.compile('|'.join(re.escape(k) + r'(?=/|[\s\x22\x27,\]\)}]|$)' for k in maps))
    return pattern.sub(lambda m: maps[m.group()], text)


def rewrite_python(text, maps):
    tree = ast.parse(text)
    constants = {'ROOT': str(ROOT)}

    def evaluate(node, mapped=False):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            value = constants.get(node.id)
            if mapped and value is not None and value.startswith(str(ROOT) + '/'):
                return str(ROOT / relocate(str(Path(value).relative_to(ROOT)), maps))
            return value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Path' and len(node.args) == 1:
            return evaluate(node.args[0], mapped)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = evaluate(node.left, mapped), evaluate(node.right, mapped)
            return str(Path(left) / right) if left is not None and right is not None else None
        return None

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                value = evaluate(node.value)
                if value is not None:
                    constants[name] = value
    lines = text.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)

    def position(line, byte_column):
        prefix = lines[line - 1].encode()[:byte_column].decode()
        return offsets[line - 1] + len(prefix)

    replacements = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        value = evaluate(node)
        if value is None or not value.startswith(str(ROOT) + '/'):
            continue
        relative = str(Path(value).relative_to(ROOT))
        new = relocate(relative, maps)
        if new != relative:
            if evaluate(node, mapped=True) == str(ROOT / new):
                continue
            left = evaluate(node.left, mapped=True)
            if left and isinstance(node.right, ast.Constant) and (ROOT / new).is_relative_to(left):
                target = str((ROOT / new).relative_to(left))
                part = node.right
                replacements.append((position(part.lineno, part.col_offset), position(part.end_lineno, part.end_col_offset), repr(target)))
            else:
                replacements.append((position(node.lineno, node.col_offset), position(node.end_lineno, node.end_col_offset), "ROOT / " + repr(new)))
    selected = []
    for start, end, new in sorted(replacements, key=lambda x: (x[0], -x[1])):
        if not selected or start >= selected[-1][1]:
            selected.append((start, end, new))
    for start, end, new in reversed(selected):
        text = text[:start] + new + text[end:]
    return replace_paths(text, maps)


def all_files():
    roots = [ROOT / 'revision_audit', ROOT / 'data/ChuanDian/review_v2', ROOT / 'data/DATA_LAYOUT.md']
    roots += [ROOT / 'experiments' / k for k in ['review_v2', 'review_v2_smoke', 'review_v3', *GROUPS, 'review_v4_smoke']]
    files = set()
    for root in roots:
        if root.is_file():
            files.add(root)
        elif root.exists():
            files.update(p for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    files.update(ROOT / p for p in ['src/data/event_pipeline.py', 'tests/data/test_review_protocol.py'])
    return sorted(files)


def plan():
    assert not any(p.is_file() and p != RECORD / 'migration_plan.json' for p in DEST.rglob('*')), DEST
    maps = make_maps()
    inventory, editable, protected = [], {}, []
    for p in all_files():
        old = str(p.relative_to(ROOT))
        new = relocate(old, maps)
        if old != new:
            assert not (ROOT / new).exists(), new
        row = {'old': old, 'new': new, 'sha256': file_digest(p), 'size': p.stat().st_size}
        inventory.append(row)
        is_current = old.startswith(tuple('experiments/' + g + '/' for g in GROUPS))
        historical = any(x in Path(old).parts for x in ['preparation_archive']) or 'calibration_pilot' in old or 'calibration_attempt' in old
        if old.startswith('revision_audit/mc_20261102/') and p.suffix == '.json':
            is_current = True
        is_script = old.startswith('revision_audit/') and p.name in SCRIPTS and p.parent.name == 'revision_audit'
        is_source = old in ['src/data/event_pipeline.py', 'tests/data/test_review_protocol.py']
        if is_source or is_script or (is_current and not historical and p.suffix in {'.yaml', '.json', '.csv'}):
            original = p.read_text()
            updated = rewrite_python(original, maps) if p.suffix == '.py' else replace_paths(original, maps)
            if old == 'src/data/event_pipeline.py':
                expected = '        cache_dir = f"{cache_namespace}/{task_prefix}"\n'
                assert updated.count(expected) == 1
                updated = updated.replace(expected, '')
            if old == 'tests/data/test_review_protocol.py':
                updated = updated.replace("tmp_path / 'review_v2/classifier'", "tmp_path / 'classifier'")
            if p.name == 'run_review_baselines.py' and is_script:
                updated = updated.replace("'experiments/review_v2'", "'experiments/revision/main'")
                updated = updated.replace("'experiments/revision/archive/review_v2'", "'experiments/revision/main'")
                updated = updated.replace("'baselines'", "'runs/references'").replace("'baselines/regression'", "'runs/references/regression'")
            if p.name == 'evaluate_full_magnitude_test.py' and is_script:
                updated = updated.replace("'" + ARCHIVE + "'", "'scripts/analyze'")
            if updated != original or p.suffix == '.json':
                editable[new] = {'old': old, 'original': original, 'updated': updated, 'path_updated': updated}
        if p.suffix in {'.pth', '.pt', '.pkl', '.npz', '.joblib'} or p.name.startswith('metrics_'):
            protected.append(row)
    # Refresh hashes only when they refer to exact pre-migration bytes. Historical originals are backed up.
    hashes = {}
    for _ in range(12):
        hashes = {digest(e['original'].encode()): digest(e['updated'].encode()) for e in editable.values() if e['original'] != e['updated']}
        changed = False
        for path, e in editable.items():
            if not path.endswith('.json'):
                continue
            def values(node):
                if isinstance(node, dict):
                    return {k: values(v) for k, v in node.items()}
                if isinstance(node, list):
                    return [values(v) for v in node]
                return hashes.get(node, node) if isinstance(node, str) else node
            before = json.loads(e['updated'])
            after = values(json.loads(e['path_updated']))
            if before != after:
                e['updated'] = json.dumps(after, indent=2) + '\n'
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError('Hash dependencies did not converge')
    edits = []
    for path, e in editable.items():
        if e['original'] == e['updated']:
            continue
        if path.endswith('.py'):
            ast.parse(e['updated'])
        elif path.endswith('.yaml'):
            import yaml
            assert yaml.safe_load(e['updated']) == yaml.safe_load(replace_paths(e['original'], maps)), path
        edits.append({'old': e['old'], 'new': path, 'before_sha256': digest(e['original'].encode()),
                      'after_sha256': digest(e['updated'].encode()), 'content': e['updated']})
    # Preserve the canonical catalogs and original raw bytes outside the moved trees.
    old_plan = json.loads((ROOT / 'revision_audit/data_layout_20261104/plan.json').read_text())
    for row in old_plan['files']:
        protected.append({'old': str(Path(row['new']).relative_to(ROOT)), 'new': str(Path(row['new']).relative_to(ROOT)),
                          'sha256': row['sha256'], 'size': row['size']})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'mappings': maps,
              'files': inventory, 'protected': protected, 'edits': edits,
              'remove_data_links': ['data/review_v3', 'data/review_v4']}
    RECORD.mkdir(parents=True, exist_ok=True)
    save(RECORD / 'migration_plan.json', report)
    print(json.dumps({'files': len(inventory), 'edits': len(edits), 'immutable_files': len(protected), 'maps': len(maps)}), flush=True)


def apply():
    report = json.loads((RECORD / 'migration_plan.json').read_text())
    for row in report['files']:
        assert file_digest(ROOT / row['old']) == row['sha256'], row['old']
    backup = DEST / 'archive/layout_before_20261104.tar.gz'
    backup.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(backup, 'x:gz') as archive:
        for row in report['edits']:
            archive.add(ROOT / row['old'], arcname=row['old'])
        for name in report['remove_data_links']:
            assert (ROOT / name).is_symlink(), name
            archive.add(ROOT / name, arcname=name)
        archive.add(ROOT / 'revision_audit/data_layout_20261104/compatibility',
                    arcname='revision_audit/data_layout_20261104/compatibility')
    # Only unlink verified aliases; no directory traversal or recursive deletion.
    compatibility = ROOT / 'revision_audit/data_layout_20261104/compatibility'
    for p in sorted(compatibility.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if p.is_symlink():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
        else:
            raise RuntimeError('Unexpected payload in compatibility directory: ' + str(p))
    compatibility.rmdir()
    for name in report['remove_data_links']:
        (ROOT / name).unlink()
    for row in report['files']:
        old, new = ROOT / row['old'], ROOT / row['new']
        if old != new:
            new.parent.mkdir(parents=True, exist_ok=True)
            assert not new.exists(), new
            old.rename(new)
    for row in report['edits']:
        path = ROOT / row['new']
        assert file_digest(path) == row['before_sha256'], row['new']
        path.write_text(row['content'])
        assert file_digest(path) == row['after_sha256'], row['new']
    old_roots = [ROOT / 'revision_audit', ROOT / 'data/ChuanDian/review_v2']
    old_roots += [ROOT / 'experiments' / k for k in ['review_v2', 'review_v2_smoke', 'review_v3', *GROUPS, 'review_v4_smoke']]
    for base in old_roots:
        if not base.exists():
            continue
        for parent, dirs, files in os.walk(base, topdown=False):
            Path(parent).rmdir()
    verify(report)


def verify(report=None):
    report = report or json.loads((RECORD / 'migration_plan.json').read_text())
    edits = {row['new']: row['after_sha256'] for row in report['edits']}
    relocated = {}
    followup = RECORD / 'followup_changes.json'
    if followup.exists():
        extra = json.loads(followup.read_text())['changes']
        relocated = {row['old']: row['new'] for row in json.loads(followup.read_text()).get('relocations', [])}
        edits.update({row['path']: row['after_sha256'] for row in extra})
        for row in extra:
            assert file_digest(ROOT / row['path']) == row['after_sha256'], row['path']
    for row in report['files']:
        path = relocated.get(row['new'], row['new'])
        assert file_digest(ROOT / path) == edits.get(row['new'], row['sha256']), path
    for row in report['protected']:
        assert file_digest(ROOT / row['new']) == row['sha256'], row['new']
    for name in report['remove_data_links']:
        assert not os.path.lexists(ROOT / name), name
    assert not (ROOT / 'revision_audit').exists()
    assert not list((ROOT / 'experiments').glob('review_v*'))
    result = {'files_verified': len(report['files']), 'immutable_files_verified': len(report['protected']),
              'documented_path_edits': len(report['edits']), 'old_data_links_removed': True,
              'old_experiment_roots_removed': True, 'old_audit_root_removed': True}
    save(RECORD / 'verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['plan', 'apply', 'verify'])
    mode = parser.parse_args().mode
    {'plan': plan, 'apply': apply, 'verify': verify}[mode]()
