#!/usr/bin/env python3
"""Index dataset inputs, experiment configs, runs and result files without training."""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import yaml


def read_config(path):
    value = json.loads(path.read_text()) if path.suffix == '.json' else yaml.safe_load(path.read_text())
    return value if isinstance(value, dict) else {}


def experiment_dirs(base):
    """Traverse study directories, never the contents of archived run trees."""
    seen = set()

    def walk(parent, depth=0):
        if depth > 6 or parent.resolve() in seen:
            return
        seen.add(parent.resolve())
        names = {p.name for p in parent.iterdir()}
        if 'runs' in names or ('configs' in names and ('reports' in names or 'plan.json' in names)):
            yield parent
            return
        for p in sorted(parent.iterdir()):
            if p.is_dir() and p.name not in {'archive', 'registry', 'reports', '__pycache__', 'raw', 'processed'}:
                yield from walk(p, depth + 1)

    yield from walk(base)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_index(root, output):
    datasets = []
    for p in sorted((root / 'data').iterdir()):
        if not p.is_dir():
            continue
        row = {'dataset': p.name, 'path': str(p.relative_to(root)), 'storage_target': str(p.resolve()),
               'raw': [], 'processed': [], 'cache': str(p / 'cache/windows'), 'catalogs': str(p / 'catalogs')}
        for kind in ('raw', 'processed'):
            folder = p / kind
            if folder.is_dir():
                row[kind] = [str(c.relative_to(root)) for c in sorted(folder.iterdir()) if c.is_file() or c.is_symlink()]
        datasets.append(row)

    groups, runs, errors = [], [], []
    studies = list(experiment_dirs(root / 'experiments'))
    if (root / 'checkpoints').is_dir():
        studies.append(root / 'checkpoints')
    for p in studies:
        standalone = p == root / 'checkpoints'
        group = 'standalone' if standalone else str(p.relative_to(root / 'experiments'))
        group_datasets = set()
        configs_dir, runs_dir, reports_dir = p / 'configs', p if standalone else p / 'runs', p / 'reports'
        group_configs = []
        if configs_dir.exists():
            group_configs = sorted(c for c in configs_dir.iterdir() if c.suffix in {'.yaml', '.yml', '.json'})
        config_candidates = [(c.parent, c) for c in group_configs]
        if runs_dir.is_dir():
            for run in sorted(runs_dir.iterdir()):
                if not run.is_dir():
                    continue
                cfg_path = next((run / name for name in ('config.yaml', 'config_input.yaml', 'config.json') if (run / name).is_file()), None)
                if cfg_path:
                    config_candidates.append((run, cfg_path))
        for run, cfg_path in config_candidates:
            try:
                cfg = read_config(cfg_path)
            except Exception as exc:
                errors.append({'path': str(cfg_path), 'error': str(exc)})
                continue
            dataset = str(cfg.get('dataset', ''))
            if not dataset:
                continue
            group_datasets.add(dataset)
            if run == configs_dir:
                continue
            base = root / 'data' / dataset
            source = sorted((base / 'raw').glob('*.csv'))
            selected = source if len(source) == 1 else [s for s in source if s.name.startswith('processed_')]
            model_cfg = cfg.get('mixer_model_config') or {}
            runs.append({'experiment': group, 'run': str(run.relative_to(root)), 'config': str(cfg_path.relative_to(root)),
                         'dataset': dataset, 'model': cfg.get('model', ''), 'seed': cfg.get('seed', ''),
                         'input_csv': str(selected[0].relative_to(root)) if len(selected) == 1 else '',
                         'catalog_cfg': json.dumps(cfg.get('catalog_cfg', {}), sort_keys=True, default=str),
                         'split_dates': json.dumps(cfg.get('split_dates', {}), sort_keys=True, default=str),
                         'Mc': cfg.get('Mc', ''), 'Mf': cfg.get('Mf', ''), 'Twindow': cfg.get('Twindow', ''), 'Tfore': cfg.get('Tfore', ''),
                         'backbone': model_cfg.get('ssm_cfg', {}).get('layer', '') if isinstance(model_cfg, dict) and isinstance(model_cfg.get('ssm_cfg'), dict) else '',
                         'metrics': ';'.join(str(f.relative_to(root)) for f in sorted(run.glob('metrics*.json'))),
                         'checkpoints': ';'.join(str(f.relative_to(root)) for f in sorted(run.glob('*model*.pth'))),
                         'storage_target': str(run.resolve())})
        reports = sorted(str(c.relative_to(root)) for c in reports_dir.glob('*.csv')) if reports_dir.is_dir() else []
        groups.append({'experiment': group, 'datasets': sorted(group_datasets),
                       'configs': [str(c.relative_to(root)) for c in group_configs], 'reports': reports,
                       'run_count': sum(r['experiment'] == group for r in runs), 'storage_target': str(p.resolve())})

    notebook_models = []
    for spec_file in sorted((root / 'notebooks/config/model_specs').rglob('*.json')):
        try:
            spec = json.loads(spec_file.read_text())
        except Exception as exc:
            errors.append({'path': str(spec_file), 'error': str(exc)})
            continue
        for config_name, config in spec.get('configs', {}).items():
            for model in config.get('models', []):
                checkpoint = model.get('checkpoint_rel_path', '')
                if checkpoint:
                    weight = root / checkpoint
                    materialized = False
                    if weight.is_file():
                        with weight.open('rb') as stream:
                            materialized = not stream.read(64).startswith(b'version https://git-lfs.github.com/spec/v1')
                    notebook_models.append({'config_file': str(spec_file.relative_to(root)), 'config_name': config_name,
                                            'label': model.get('label', ''), 'checkpoint': checkpoint,
                                            'exists': weight.is_file(), 'materialized': materialized})
    output.mkdir(parents=True, exist_ok=True)
    (output / 'datasets.json').write_text(json.dumps(datasets, indent=2) + '\n')
    (output / 'experiments.json').write_text(json.dumps(groups, indent=2) + '\n')
    (output / 'notebook_models.json').write_text(json.dumps(notebook_models, indent=2) + '\n')
    fields = ['experiment', 'run', 'config', 'dataset', 'model', 'seed', 'input_csv', 'catalog_cfg', 'split_dates', 'Mc', 'Mf', 'Twindow', 'Tfore', 'backbone', 'metrics', 'checkpoints', 'storage_target']
    write_csv(output / 'runs.csv', runs, fields)
    (output / 'errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    text = ['# Data and experiment index', '',
            'Regenerate with `python scripts/maintenance/index_experiments.py`.', '',
            '`datasets.json` lists source and processed inputs. `runs.csv` maps each dataset to its exact run, YAML, metrics and checkpoints.', '',
            '`notebook_models.json` maps each notebook preset (including `SCEDC_t` and `ChuanDian_t`) to the exact checkpoint and records whether its weights are materialized.', '',
            'SWC is the paper name for `ChuanDian`; SCSN is the paper name for `SCEDC`. Names remain unchanged in data loading.', '',
            '| Experiment | Datasets | Runs | First result file |', '| --- | --- | ---: | --- |']
    for g in groups:
        text.append('| `' + g['experiment'] + '` | ' + ', '.join(g['datasets']) + ' | ' + str(g['run_count']) + ' | `' + (g['reports'][0] if g['reports'] else '') + '` |')
    (output / 'README.md').write_text('\n'.join(text) + '\n')
    return datasets, groups, runs, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dataset')
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / 'experiments/registry'
    datasets, groups, runs, errors = build_index(root, output)
    print(json.dumps({'datasets': len(datasets), 'experiments': len(groups), 'runs': len(runs), 'errors': len(errors), 'output': str(output)}))
    if args.dataset:
        selected = [r for r in runs if r['dataset'] == args.dataset]
        counts = Counter(r['experiment'] for r in selected)
        for group, count in sorted(counts.items()):
            sample = next(r for r in selected if r['experiment'] == group)
            print(json.dumps({'experiment': group, 'runs': count, 'sample_config': sample['config'], 'input_csv': sample['input_csv']}))


if __name__ == '__main__':
    main()
