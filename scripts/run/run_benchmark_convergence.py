"""Validation-only convergence checks for regional RECAST benchmark baselines."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

BASE = Path('/root/autodl-fs/em_eqf_training/unified_tpp_20261106')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def runtime(root):
    project = root / 'project'
    sys.path[:0] = [str(project), str(project / 'scripts/run')]
    import torch
    import yaml
    from run_expanded_multiregion import Study, digest, spans
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    return torch, yaml, Study, digest, spans


def prepare(args):
    root, reference = args.root.resolve(), args.reference.resolve()
    root.mkdir(parents=True, exist_ok=False)
    project = args.source_project.resolve() if args.source_project else reference / 'project'
    (root / 'project').symlink_to(project, target_is_directory=True)
    shutil.copy2(Path(__file__), root / 'recast_driver.py')
    previous = read(reference / 'plan.json')
    comparison = read(reference / 'gr/comparison.json')
    targets = {}
    for target, cohort in previous['cohorts'].items():
        targets[target] = dict(steps=800 * (len(cohort['sources']) + 1),
                               sources=len(cohort['sources']),
                               data=str(reference / 'data' / target),
                               manifest_sha256=sha(reference / 'data' / target / 'dataset.json'))
    critical = ['scripts/run/run_expanded_multiregion.py', 'src/models/tpp/region_recast.py',
                'src/models/tpp/region_mixer_tpp.py', 'src/models/tpp/recurrent/model_v1.py',
                'src/data/unified_tpp.py', 'src/train/scheduler.py']
    write(root / 'plan.json', dict(reference=str(reference), targets=targets,
          rates=[.0003, .001, .003], seeds=[0, 1, 2], steps_rule='800*(source_regions+1)',
          mode=args.mode, history=4096, train_events=2048, validate_every=100,
          selection='seed 0 full validation only, including the existing 800-step baseline',
          driver_sha256=sha(root / 'recast_driver.py'),
          source_files={f: sha(root / 'project' / f) for f in critical},
          reference_sha256=sha(reference / 'gr/comparison.json')))
    assert len(comparison['rows']) == 54
    print('PREPARED', str(root), flush=True)


def verify(root, plan):
    assert sha(root / 'recast_driver.py') == plan['driver_sha256']
    for name, value in plan['source_files'].items():
        assert sha(root / 'project' / name) == value, name
    assert sha(Path(plan['reference']) / 'gr/comparison.json') == plan['reference_sha256']


def case_name(target, seed, lr):
    return f'{target}__s{seed}__lr{lr:g}'


def study_for(root, plan, target, seed, lr, helpers):
    _, yaml, Study, digest, _ = helpers
    reference = Path(plan['reference'])
    config = yaml.safe_load((reference / 'gr' / target / f's{seed}' / 'transfer.yaml').read_text())
    case = root / 'recast' / case_name(target, seed, lr)
    config['paths'].update(runs=str(case / 'runs'), reports=str(case / 'shared/reports'))
    config['training'].update(seeds=[seed], target_steps=plan['targets'][target]['steps'],
                              target_lrs=[lr], target_validate_every=100)
    case.mkdir(parents=True, exist_ok=True)
    (case / 'shared/reports').mkdir(exist_ok=True)
    (case / 'runs').mkdir(exist_ok=True)
    (case / 'transfer.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    study = Study(config)
    study.plan.update(neural_fits=1, scope=plan.get('mode', 'local') + '_only_convergence')
    study.plan_id = digest(study.plan)
    write(case / 'study_plan.json', dict(plan_id=study.plan_id, **study.plan))
    return case, study


def worker(args):
    root = args.root.resolve()
    plan = read(root / 'plan.json')
    verify(root, plan)
    helpers = runtime(root)
    torch, _, _, digest, spans = helpers
    target, seed, lr = args.target, args.seed, args.lr
    case, study = study_for(root, plan, target, seed, lr, helpers)
    data = study.load('development.pt')
    region = data['regions'][target]
    pool = spans(region['train_start'], region['train_end'], 2048)
    rng = random.Random(seed * 100003 + data['vocabulary'][target] * 1009)
    draws = rng.choices(pool, weights=[count for _, count in pool], k=study.options['target_steps'])
    reference = Path(plan['reference'])
    mode = plan.get('mode', 'local')
    original_runs = reference / 'gr' / target / f's{seed}' / 'runs'
    old = read(original_runs / f'{target}__{mode}__s{seed}__lr0.0003' / 'result.json')
    assert digest({target: draws[:800]}) == old['draw_sha256']
    parent = None
    if mode == 'fine':
        parent_folder = original_runs / f'{target}__source__s{seed}__lr0.0003'
        parent = read(parent_folder / 'result.json')
        assert sha(parent_folder / 'best_model.pth') == parent['checkpoint_sha256'] == old['parent_sha256']
        link = study.runs / parent['label']
        if not link.exists():
            link.symlink_to(parent_folder, target_is_directory=True)
        assert link.resolve() == parent_folder.resolve()
    started = time.time()
    result = study.neural_fit(data, target, mode, seed, lr, parent=parent)
    prior_timing = read(case / 'fit.json').get('fit_seconds') if (case / 'fit.json').exists() else None
    assert result['initial_sha256'] == old['initial_sha256']
    assert result['draw_sha256'] == digest({target: draws})
    if parent:
        assert result['parent_sha256'] == parent['checkpoint_sha256']
    checkpoint = study.runs / result['label'] / 'best_model.pth'
    value = dict(target=target, mode=mode, seed=seed, lr=lr, steps=study.options['target_steps'],
                 fit_seconds=prior_timing if prior_timing is not None else time.time() - started,
                 validation=result['val_total'], best_step=result['best_step'],
                 checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                 draw_prefix_sha256=old['draw_sha256'], result=result,
                 study_plan_id=study.plan_id)
    write(case / 'fit.json', value)
    if args.phase == 'score':
        assert (root / 'recast_selection.json').exists()
        selection = read(root / 'recast_selection.json')[target]
        assert selection['setting'] == 'long' and selection['lr'] == lr
        model = study._build_model(seed, data['vocabulary'])
        study.load_weights(model, result)
        testing = study.load('test.pt')
        region = testing['regions'][target]
        score, records = study.evaluate(model, testing, {target: (region['val_end'], region['test_end'])})
        value.update(test_nll=score['macro']['time'], events=region['test_end'] - region['val_end'],
                     test_records=records[target])
        write(case / 'test.json', value)
    print(json.dumps(value), flush=True)


def run_jobs(root, jobs, workers, phase):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def launch(job):
        target, seed, lr = job
        log = root / 'logs' / f'{phase}__{case_name(target, seed, lr)}.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(root / 'recast_driver.py'), phase,
                '--root', str(root), '--target', target, '--seed', str(seed), '--lr', str(lr)],
                stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
        print(f'COMPLETE {phase} {case_name(target, seed, lr)}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            future.result()


def run(args):
    root = args.root.resolve()
    plan = read(root / 'plan.json')
    verify(root, plan)
    write(root / 'recast_execution.json', dict(workers=args.workers, started_unix=time.time()))
    targets = list(plan['targets'])
    run_jobs(root, [(t, 0, lr) for t in targets for lr in plan['rates']], args.workers, 'fit')
    reference = Path(plan['reference'])
    selection = {}
    for target in targets:
        mode = plan.get('mode', 'local')
        old = read(reference / 'gr' / target / 's0/runs' /
                   f'{target}__{mode}__s0__lr0.0003/result.json')
        trials = [read(root / 'recast' / case_name(target, 0, lr) / 'fit.json') for lr in plan['rates']]
        winner = min(trials, key=lambda row: row['validation'])
        selection[target] = dict(setting='long' if winner['validation'] < old['val_total'] else 'existing800',
                                 lr=winner['lr'], new_validation=winner['validation'],
                                 old_validation=old['val_total'], candidates=trials)
    write(root / 'recast_selection.json', selection)
    jobs = [(t, s, selection[t]['lr']) for t in targets
            if selection[t]['setting'] == 'long' for s in [1, 2]]
    run_jobs(root, jobs, args.workers, 'fit')
    # All validation selection and fits finish before opening any test files.
    run_jobs(root, [(t, s, selection[t]['lr']) for t in targets
                   if selection[t]['setting'] == 'long' for s in [0, 1, 2]], min(args.workers, 8), 'score')
    previous = read(reference / 'gr/comparison.json')
    rows = []
    for target in targets:
        for seed in [0, 1, 2]:
            if selection[target]['setting'] == 'long':
                row = read(root / 'recast' / case_name(target, seed, selection[target]['lr']) / 'test.json')
            else:
                row = next(r for r in previous['rows'] if (r['target'], r['seed'], r['mode']) == (target, seed, plan.get('mode', 'local')))
                row = dict(row, setting='existing800')
            rows.append(row)
    summary = {t: dict(mean=statistics.mean(r['test_nll'] for r in rows if r['target'] == t),
                       sd=statistics.stdev(r['test_nll'] for r in rows if r['target'] == t)) for t in targets}
    write(root / 'recast_comparison.json', dict(mode=plan.get('mode', 'local'), summary=summary, selection=selection, rows=rows))
    write(root / 'recast_complete.json', dict(passed=True, targets=targets))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'fit', 'score', 'run'])
    parser.add_argument('--root', type=Path, default=BASE / 'benchmark_convergence_h4096_3seed')
    parser.add_argument('--reference', type=Path, default=BASE / 'cross_recast_benchmarks_h4096_3seed')
    parser.add_argument('--workers', type=int, default=18)
    parser.add_argument('--mode', choices=['local', 'fine'], default='local')
    parser.add_argument('--source-project', type=Path)
    parser.add_argument('--target')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=.0003)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'fit': worker, 'score': worker}[args.phase](args)
