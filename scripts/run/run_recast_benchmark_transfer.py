"""Evaluate regional RECAST transfer on six chronologically held-out benchmarks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import torch
import yaml

from run_expanded_multiregion import ROOT, Study, make_batch

BASE = Path('/root/autodl-fs/em_eqf_training/unified_tpp_20261106')
TARGETS = ('QTM_SaltonSea_10', 'QTM_SanJacinto_10', 'SCEDC_25',
           'ComCat_25', 'WHITE_06', 'ChuanDian_30')
FIELDS = ('ns', 'physical_ns', 'magnitude', 'raw_magnitude', 'event_id',
          'latitude', 'longitude', 'mag_type')
ARMS = ('gr', 'center')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(folder, filename):
    manifest = read(folder / 'dataset.json')
    assert sha(folder / filename) == manifest['artifacts'][filename]
    return torch.load(folder / filename, map_location='cpu', weights_only=False)


def prepare(root):
    if (root / 'plan.json').exists():
        raise FileExistsError('Prepared cohorts already exist')
    source_dir = BASE / 'data/multiregion'
    source = load(source_dir, 'development.pt')
    folders = {t: BASE / 'data/benchmark' for t in TARGETS}
    folders.update({t: BASE / 'public_qtm_chuandian_h4096_3seed/data'
                    for t in TARGETS if t.startswith(('QTM_', 'ChuanDian'))})
    datasets = {p: (load(p, 'development.pt'), load(p, 'test.pt')) for p in set(folders.values())}
    vocabulary = {name: i for i, name in enumerate(sorted(set(TARGETS) | set(source['regions'])))}
    base_config = yaml.safe_load((BASE / 'cross_baselines_film_h4096_s0/recast/transfer.yaml').read_text())
    cohorts = {}
    for target in TARGETS:
        development, testing = datasets[folders[target]]
        original = development['regions'][target]
        end = original['time_bounds_ns']['train_end']
        # Reserve six calendar months immediately preceding the target training boundary.
        month = np.datetime64(end, 'ns').astype('datetime64[M]') - np.timedelta64(6, 'M')
        cutoff = int(month.astype('datetime64[ns]').astype(np.int64))
        regions, exclusions = {}, {}
        for name, original_source in source['regions'].items():
            if name == 'NCEDC' and target != 'ChuanDian_30':
                exclusions[name] = 'California source excluded from California targets'
                continue
            lo = original_source['train_start']
            mid, hi = map(int, np.searchsorted(original_source['ns'], [cutoff, end]))
            if mid - lo < 128 or hi - mid < 32:
                exclusions[name] = 'Insufficient chronological source training or validation events'
                continue
            region = copy.deepcopy(original_source)
            for key in FIELDS:
                region[key] = region[key][:hi]
            region.update(source_train_end=mid, train_end=hi, val_end=hi, test_end=hi,
                          mag_mean_train=float(region['magnitude'][:mid].mean()))
            region['time_bounds_ns'].update(source_train_end=cutoff, train_end=end, val_end=end, test_end=end)
            assert region['ns'][255] < cutoff and region['ns'][-1] < end
            regions[name] = region
        assert len(regions) >= 2, (target, exclusions)
        target_dev = copy.deepcopy(original)
        target_dev['source_train_end'] = target_dev['train_end']
        target_dev['time_bounds_ns']['source_train_end'] = end
        mean = float(target_dev['magnitude'][:target_dev['train_end']].mean())
        target_dev['mag_mean_train'] = mean
        regions[target] = target_dev
        target_test = copy.deepcopy(testing['regions'][target])
        target_test.update(mag_mean_train=mean, source_train_end=target_test['train_end'])
        target_test['time_bounds_ns']['source_train_end'] = end
        test_regions = dict(regions, **{target: target_test})
        directory = root / 'data' / target
        directory.mkdir(parents=True, exist_ok=True)
        for filename, values in [('development.pt', regions), ('test.pt', test_regions)]:
            torch.save(dict(regions=values, vocabulary=vocabulary, calibration_events=256), directory / filename)
        counts = {name: dict(train=r['train_end'] - r['train_start'],
                            source_train=r['source_train_end'] - r['train_start'],
                            source_validation=r['train_end'] - r['source_train_end'])
                  for name, r in regions.items()}
        manifest = dict(schema='unified-tpp-v1', group='multiregion', regions=counts,
                        calibration_events=256, artifacts={f: sha(directory / f) for f in ('development.pt', 'test.pt')})
        write(directory / 'dataset.json', manifest)
        sources = sorted(set(regions) - {target})
        cohorts[target] = dict(sources=sources, excluded=exclusions, source_train_end_ns=cutoff,
                               source_validation_end_ns=end, counts=counts,
                               original_manifest_sha256=sha(folders[target] / 'dataset.json'),
                               original_data=str(folders[target]), cohort_manifest_sha256=sha(directory / 'dataset.json'))
        for arm in ARMS:
            for seed in (0, 1, 2):
                job = root / arm / target / f's{seed}'
                job.mkdir(parents=True, exist_ok=True)
                cfg = copy.deepcopy(base_config)
                cfg['paths'].update(data=str(directory), runs=str(job / 'runs'), reports=str(job / 'shared/reports'))
                cfg['training'].update(targets=[target], seeds=[seed], baseline='recast',
                                       recurrent_width=32, region_condition_stats=True,
                                       region_magnitude_preprocess=arm, driver_sha256=sha(Path(__file__)))
                (job / 'transfer.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(target, 'sources', len(sources), 'source events', sum(counts[n]['source_train'] for n in sources), flush=True)
    write(root / 'plan.json', dict(cohorts=cohorts, seeds=[0, 1, 2], arms=ARMS,
          driver_sha256=sha(Path(__file__)), source_manifest_sha256=sha(source_dir / 'dataset.json'),
          magnitude_center_statistics='Mean of each training sequence, including its auxiliary prefix; source means stop before source validation'))


def preflight(root):
    plan = read(root / 'plan.json')
    assert plan['driver_sha256'] == sha(Path(__file__))
    results = []
    for target in TARGETS:
        for arm in ARMS:
            cfg = yaml.safe_load((root / arm / target / 's0/transfer.yaml').read_text())
            study = Study(cfg)
            data = study.load('development.pt')
            region = data['regions'][target]
            first = region['train_start']
            count = min(128, region['train_end'] - first)
            batch = make_batch(data, target, [(first, count)], 4096, study.device)
            model = study._build_model(0, data['vocabulary'])
            loss = model.nll_loss(batch, reduction='per_event')['time'].mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert int(batch.event_index[batch.nll_event_mask.bool()].min()) == first
            results.append(dict(target=target, arm=arm, training_start=first, loss=float(loss.detach())))
            del model, batch
    write(root / 'preflight.json', dict(passed=True, rows=results))


def run_job(root, arm, phase, target, seed):
    job = root / arm / target / f's{seed}'
    command = [sys.executable, '-u', str(ROOT / 'scripts/run/run_expanded_multiregion.py'),
               '--config', str(job / 'transfer.yaml'), '--stage', phase]
    with (job / f'{phase}.log').open('ab') as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    print(arm, phase, target, seed, code, flush=True)
    if code:
        raise RuntimeError(f'{job}/{phase}.log failed with {code}')


def aggregate(root, arm):
    rows = []
    old = read(BASE / 'public_qtm_chuandian_h4096_3seed/comparison.json')
    old_rows = {(r['target'], r['seed']): r for r in old['rows'] if r['model'] == 'recast'}
    for target in TARGETS:
        for seed in (0, 1, 2):
            job = root / arm / target / f's{seed}'
            scores = read(job / 'reports/scores.json')
            manifest = read(job / 'runs/plan.json')
            assert manifest['plan_id'] == scores['plan_id']
            for row in scores['checkpoints']:
                fit = read(job / 'runs' / row['label'] / 'result.json')
                assert sha(job / 'runs' / row['label'] / 'best_model.pth') == row['checkpoint_sha256']
                assert row['events'] == old_rows[target, seed]['test_events']
                assert len(row['windows']) == 1 and row['windows'][0]['history_events'] == row['windows'][0]['first']
                rows.append(dict(target=target, seed=seed, mode=row['mode'], test_nll=row['time'],
                                 events=row['events'], best_step=row['best_step'], draw_sha256=fit['draw_sha256'],
                                 checkpoint_sha256=row['checkpoint_sha256']))
    summary = {target: {mode: dict(mean=statistics.mean(r['test_nll'] for r in rows if r['target'] == target and r['mode'] == mode),
                                   sd=statistics.stdev(r['test_nll'] for r in rows if r['target'] == target and r['mode'] == mode))
                        for mode in ('local', 'zero', 'fine')} for target in TARGETS}
    legacy = {target: old['summary'][target]['recast'] for target in TARGETS}
    etas = read(root / 'etas/comparison.json')['summary']
    write(root / arm / 'comparison.json', dict(protocol='full_sequence_calendar', arm=arm,
          test_metric='per-event time NLL', summary=summary, legacy_single_region=legacy, etas=etas, rows=rows))
    with (root / arm / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def etas_job(root, target, seed):
    data = read(root / 'plan.json')['cohorts'][target]['original_data']
    folder = root / 'etas' / target / f's{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'model'
    entry = str(ROOT / 'scripts/run/run_unified_tpp.py')
    fitting = [sys.executable, '-u', entry, 'fit', '--data', data, '--output', str(output),
               '--target', target, '--model', 'etas', '--seed', str(seed), '--lr', '.001',
               '--steps', '800', '--train-events', '2048', '--validate-every', '100']
    testing = [sys.executable, '-u', entry, 'test', '--data', data, '--output', str(output)]
    for phase, command in [('fit', fitting), ('test', testing)]:
        if (output / ('complete.json' if phase == 'fit' else 'test.json')).exists():
            continue
        with (folder / f'{phase}.log').open('ab') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    value = read(output / 'test.json')
    print('ETAS', target, seed, value['time_nll_per_event'], flush=True)
    return dict(target=target, seed=seed, test_nll=value['time_nll_per_event'], events=value['events'])


def etas_baselines(root):
    existing = read(BASE / 'benchmark_3seed/comparison_3seed.json')['summary']
    targets = [t for t in TARGETS if t not in existing or 'etas' not in existing[t]]
    plan = read(root / 'plan.json')
    for target in set(TARGETS) - set(targets):
        original = read(Path(plan['cohorts'][target]['original_data']) / 'dataset.json')
        assert existing[target]['etas']['events'] == original['regions'][target]['counts']['test']
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda job: etas_job(root, *job), [(t, s) for t in targets for s in (0, 1, 2)]))
    summary = {t: existing[t]['etas'] for t in TARGETS if t not in targets}
    for target in targets:
        values = [r['test_nll'] for r in rows if r['target'] == target]
        summary[target] = dict(mean=statistics.mean(values), sd=statistics.stdev(values))
    write(root / 'etas/comparison.json', dict(summary=summary, rows=rows))


def run(root, workers):
    preflight(root)
    jobs = [(t, s) for t in TARGETS for s in (0, 1, 2)]
    with ThreadPoolExecutor(max_workers=1) as background:
        etas = background.submit(etas_baselines, root)
        for arm in ARMS:
            for phase in ('fit', 'score'):
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(lambda job: run_job(root, arm, phase, *job), jobs))
        etas.result()
        for arm in ARMS:
            aggregate(root, arm)
    gr, center = (read(root / arm / 'comparison.json') for arm in ARMS)
    assert [r['draw_sha256'] for r in gr['rows']] == [r['draw_sha256'] for r in center['rows']]
    write(root / 'comparison.json', dict(gr=gr, center=center, paired_training_draws=True))
    write(root / 'complete.json', dict(passed=True, recurrent_fits=108,
          new_etas_fits=len(read(root / 'etas/comparison.json')['rows']), targets=TARGETS, seeds=[0, 1, 2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=18)
    options = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    if options.stage == 'prepare':
        prepare(options.output)
    else:
        run(options.output, options.workers)
