#!/usr/bin/env python3
"""Index dataset inputs, experiment configs, runs and result files without training."""
import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import yaml

CONFIG_NAMES = ('config.yaml', 'config_input.yaml', 'config.json')
SKIP_DIRS = {'archive', 'registry', 'reports', 'configs', '__pycache__', 'tensorboard', 'raw', 'processed', 'cache', 'figures', 'plots'}


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
        if 'runs' in names or 'optuna_trials' in names or any(n in names for n in CONFIG_NAMES) or ('configs' in names and ('reports' in names or 'plan.json' in names)):
            yield parent
            return
        for p in sorted(parent.iterdir()):
            if p.name not in SKIP_DIRS and p.is_dir():
                yield from walk(p, depth + 1)

    yield from walk(base)


def run_configs(base):
    """Read each run directory once, including nested Optuna trials."""
    seen = set()

    def walk(folder, depth=0):
        resolved = folder.resolve()
        if depth > 8 or resolved in seen:
            return
        seen.add(resolved)
        entries = {p.name: p for p in folder.iterdir()}
        config = next((entries[n] for n in CONFIG_NAMES if n in entries and entries[n].is_file()), None)
        if config:
            yield folder, config, entries
        for name, path in sorted(entries.items()):
            if name in SKIP_DIRS:
                continue
            if config and name not in {'optuna_trials', 'runs', 'profiles'}:
                continue
            if path.is_dir():
                yield from walk(path, depth + 1)

    if base.is_dir():
        yield from walk(base)


def write_csv(path, rows, fields):
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.index-') as folder:
        temporary = Path(folder) / path.name
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)


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
    source_csvs = {}
    studies = list(experiment_dirs(root / 'experiments'))
    if (root / 'checkpoints').is_dir():
        studies.append(root / 'checkpoints')
    for p in studies:
        standalone = p == root / 'checkpoints'
        group = 'standalone' if standalone else str(p.relative_to(root / 'experiments'))
        group_datasets = set()
        configs_dir, reports_dir = p / 'configs', p / 'reports'
        runs_dir = p if standalone or not (p / 'runs').is_dir() else p / 'runs'
        group_configs = []
        if configs_dir.exists():
            group_configs = sorted(c for c in configs_dir.iterdir() if c.suffix in {'.yaml', '.yml', '.json'})
        config_candidates = [(c.parent, c, None) for c in group_configs]
        config_candidates.extend(run_configs(runs_dir))
        first_run = len(runs)
        for run, cfg_path, entries in config_candidates:
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
            if dataset not in source_csvs:
                source = sorted((base / 'raw').glob('*.csv'))
                source_csvs[dataset] = source if len(source) == 1 else [s for s in source if s.name.startswith('processed_')]
            selected = source_csvs[dataset]
            model_cfg = cfg.get('mixer_model_config') or {}
            runs.append({'experiment': group, 'run': str(run.relative_to(root)), 'config': str(cfg_path.relative_to(root)),
                         'dataset': dataset, 'model': cfg.get('model', ''), 'seed': cfg.get('seed', ''),
                         'input_csv': str(selected[0].relative_to(root)) if len(selected) == 1 else '',
                         'catalog_cfg': json.dumps(cfg.get('catalog_cfg', {}), sort_keys=True, default=str),
                         'split_dates': json.dumps(cfg.get('split_dates', {}), sort_keys=True, default=str),
                         'Mc': cfg.get('Mc', ''), 'Mf': cfg.get('Mf', ''), 'Twindow': cfg.get('Twindow', ''), 'Tfore': cfg.get('Tfore', ''),
                         'backbone': model_cfg.get('ssm_cfg', {}).get('layer', '') if isinstance(model_cfg, dict) and isinstance(model_cfg.get('ssm_cfg'), dict) else '',
                         'metrics': ';'.join(str(entries[n].relative_to(root)) for n in sorted(entries) if n.startswith('metrics') and n.endswith('.json')),
                         'checkpoints': ';'.join(str(entries[n].relative_to(root)) for n in sorted(entries) if 'model' in n and n.endswith('.pth')),
                         'storage_target': str(run.resolve())})
        reports = sorted(str(c.relative_to(root)) for c in reports_dir.glob('*.csv')) if reports_dir.is_dir() else []
        groups.append({'experiment': group, 'datasets': sorted(group_datasets),
                       'configs': [str(c.relative_to(root)) for c in group_configs], 'reports': reports,
                       'run_count': len(runs) - first_run, 'storage_target': str(p.resolve())})

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
            'Regenerate with `python scripts/maintenance/index_experiments.py`. Query the existing index with `--dataset ChuanDian`; add `--refresh` to rebuild first.', '',
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
    parser.add_argument('--refresh', action='store_true', help='Rebuild before a dataset query; queries otherwise read the saved index.')
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / 'experiments/registry'
    if args.dataset and not args.refresh:
        index = output / 'runs.csv'
        if not index.is_file():
            parser.error(f'No saved index at {index}; run without --dataset or add --refresh.')
        with index.open(newline='') as stream:
            runs = list(csv.DictReader(stream))
    else:
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
