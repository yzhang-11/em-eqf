"""Fit native unconstrained temporal ETAS with full-training L-BFGS and multistart."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

BASE = Path('/root/autodl-fs/em_eqf_training/unified_tpp_20261106')
STARTS = ('warm_raw', 'fresh_rate', 'fresh_p09')
LIMIT_FIELDS = ('etas_max_branching_ratio', 'etas_effective_branching_t_max',
                'etas_min_omori_p', 'etas_constraint_softness')


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
    module = importlib.util.spec_from_file_location('convergence_shared', root / 'recast_driver.py')
    shared = importlib.util.module_from_spec(module)
    module.loader.exec_module(shared)
    torch, yaml, *_ = shared.runtime(root)
    import run_unified_tpp as native
    from run_expanded_multiregion import ScoredETAS
    return torch, yaml, native, ScoredETAS


def prepare(args):
    root = args.root.resolve()
    master = read(root / 'plan.json')
    destination = root / 'etas_driver.py'
    shutil.copy2(Path(__file__), destination)
    _, yaml, *_ = runtime(root)
    config_path = Path('/root/autodl-tmp/em_eqf/experiments/revision/generation_baselines/configs/etas_scsn_main_seed_0.yaml')
    cfg = yaml.safe_load(config_path.read_text())
    cfg.update(etas_enforce_subcritical=False, etas_enforce_p_gt_one=False,
               etas_query_chunk_size=4096, etas_history_chunk_size=4096,
               etas_grad_checkpoint=True, weight_decay=0., learning_rate=1.,
               loss_reduction='per_event', scheduler_type=None, warmup_ratio=0.,
               optimizer='lbfgs', epochs=100)
    cfg.pop('catalog_cfg', None)
    for key in LIMIT_FIELDS:
        cfg.pop(key, None)
    (root / 'etas_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    critical = ['src/models/tpp/etas.py', 'src/models/tpp/common/etas_utils.py',
                'scripts/run/run_unified_tpp.py', 'src/data/unified_tpp.py']
    write(root / 'etas_plan.json', dict(reference=master['reference'], targets=master['targets'],
          starts=STARTS, seeds=[0, 1, 2], full_training_history=True, full_training_targets=True,
          enforce_p_gt_one=False, enforce_subcritical=False, weight_decay=0,
          optimizer='LBFGS strong_wolfe', max_outer=20, inner_iterations=5,
          gradient_tolerance=1e-5, selection='seed 0 validation only; historical constrained test scores excluded',
          driver_sha256=sha(destination), config_sha256=sha(root / 'etas_config.yaml'),
          source_files={f: sha(root / 'project' / f) for f in critical}))


def physical(model):
    return {key: float(getattr(model, key).detach().cpu()) for key in ['p', 'c', 'mu', 'k', 'alpha']}


def make_model(root, plan, target, seed, start, helpers):
    torch, yaml, native, ScoredETAS = helpers
    reference = Path(plan['reference'])
    folder = reference / 'etas_validation_refine' / target / f's{seed}'
    prior_meta = read(folder / 'fit_complete.json')
    cfg = yaml.safe_load((root / 'etas_config.yaml').read_text())
    cfg.update(dataset=target, Mc=prior_meta['config']['mag_completeness'],
               mag_completeness=prior_meta['config']['mag_completeness'],
               richter_b=prior_meta['config']['richter_b_mle'],
               richter_b_mle=prior_meta['config']['richter_b_mle'],
               base_rate_init=prior_meta['spec']['initial_rate'])
    model = ScoredETAS(cfg, torch.device('cuda')).native
    if start == 'warm_raw':
        # Reuse raw parameters as one initial guess; all new evaluations are free.
        model.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)['state'])
        values = physical(model)
    else:
        rate = prior_meta['spec']['initial_rate']
        b_value = cfg['richter_b_mle']
        values = dict(p=1.08, c=.1, mu=.3 * rate, k=.0073, alpha=.6 * b_value)
        if start == 'fresh_p09':
            values.update(p=.9, c=.01, mu=.7 * rate, k=.01, alpha=.85 * b_value)
        if seed:
            rng = random.Random(seed * 100003 + sum(map(ord, target)))
            values = {k: v * math.exp(rng.gauss(0, .1)) for k, v in values.items()}
    with torch.no_grad():
        for key, value in values.items():
            getattr(model, 'log_' + key).fill_(math.log(value))
    assert not model.enforce_p_gt_one and not model.enforce_subcritical
    assert torch.equal(model.p, model.log_p.exp())
    assert torch.equal(model.k, model.log_k.exp())
    return model, cfg


def verify(root, plan):
    assert sha(root / 'etas_driver.py') == plan['driver_sha256']
    assert sha(root / 'etas_config.yaml') == plan['config_sha256']
    for name, value in plan['source_files'].items():
        assert sha(root / 'project' / name) == value, name


def worker(args):
    root = args.root.resolve()
    plan = read(root / 'etas_plan.json')
    verify(root, plan)
    helpers = runtime(root)
    torch, _, native, _ = helpers
    target, seed, start = args.target, args.seed, args.start
    folder = root / 'etas' / target / f'{start}__s{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    reference = Path(plan['reference'])
    model, cfg = make_model(root, plan, target, seed, start, helpers)
    data_dir = Path(plan['targets'][target]['data'])
    manifest = read(data_dir / 'dataset.json')
    assert sha(data_dir / 'dataset.json') == plan['targets'][target]['manifest_sha256']
    assert sha(data_dir / 'development.pt') == manifest['artifacts']['development.pt']
    data = native.load_catalogs(data_dir)
    region, region_id = data['regions'][target], data['vocabulary'][target]
    dates = region['time_bounds_ns']
    spec = dict(model='etas', target=target, seed=seed, start=start, history=None,
                objective='full_training_time_nll', manifest=sha(data_dir / 'dataset.json'))
    if not (folder / 'fit.json').exists():
        batch = native.calendar_batch(region, dates['train_start'], dates['train_end'], history=None,
                                      region_id=region_id).to('cuda')
        parameters = list(model.parameters())
        optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=5, max_eval=10,
            tolerance_grad=1e-6, tolerance_change=1e-10, history_size=10, line_search_fn='strong_wolfe')
        evaluations = 0
        def closure():
            nonlocal evaluations
            evaluations += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            objective = model.nll_loss(batch, reduction='per_event').mean()
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite native ETAS objective during line search')
            objective.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Nonfinite native ETAS gradient during line search')
            return objective
        best, history = float('inf'), []
        initial = physical(model)
        started = time.time()
        status, stable = 'iteration_limit', 0
        last_train = None
        for outer in range(plan['max_outer'] + 1):
            try:
                if outer:
                    optimizer.step(closure)
                train = float(closure().detach())
            except FloatingPointError as error:
                if not (folder / 'best.pt').exists():
                    raise
                status = 'numerical_stop'
                write(folder / 'numerical_stop.json', dict(error=str(error), outer=outer))
                break
            gradient = max(float(p.grad.abs().max()) for p in parameters)
            validation = native.evaluate(model, spec, region, region_id,
                                         dates['train_end'], dates['val_end'], torch.device('cuda'))
            row = dict(outer=outer, iterations=optimizer.state[parameters[0]].get('n_iter', 0),
                       evaluations=evaluations, train_nll=train, gradient_inf=gradient,
                       validation=validation['time_nll_per_event'], parameters=physical(model),
                       seconds=time.time() - started)
            history.append(row)
            print(json.dumps(row), flush=True)
            if row['validation'] < best:
                best = row['validation']
                pending = folder / 'best.tmp'
                torch.save(dict(spec=spec, config=cfg, state=model.state_dict(),
                                validation=validation, outer=outer, train_nll=train), pending)
                pending.replace(folder / 'best.pt')
            if gradient < plan['gradient_tolerance']:
                status = 'gradient_converged'
                break
            if last_train is not None and abs(train - last_train) < 1e-8 * max(1., abs(train)):
                stable += 1
            else:
                stable = 0
            if stable >= 3:
                status = 'objective_plateau'
                break
            last_train = train
        result = dict(target=target, seed=seed, start=start, spec=spec, config=cfg,
                      initial_parameters=initial, status=status, history=history,
                      validation=best, evaluations=evaluations,
                      checkpoint=str(folder / 'best.pt'), checkpoint_sha256=sha(folder / 'best.pt'),
                      seconds=time.time() - started)
        write(folder / 'fit.json', result)
        del batch, optimizer
        torch.cuda.empty_cache()
    result = read(folder / 'fit.json')
    assert sha(folder / 'best.pt') == result['checkpoint_sha256']
    if args.phase == 'score':
        assert read(root / 'etas_selection.json')[target]['start'] == start
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state'], strict=True)
        assert sha(data_dir / 'test.pt') == manifest['artifacts']['test.pt']
        testing = native.load_catalogs(data_dir, test=True)['regions'][target]
        bounds = testing['time_bounds_ns']
        score = native.evaluate(model, spec, testing, region_id,
                                bounds['val_end'], bounds['test_end'], torch.device('cuda'))
        result.update(test=score, test_nll=score['time_nll_per_event'], parameters=physical(model))
        write(folder / 'test.json', result)
    print(json.dumps(dict(target=target, seed=seed, start=start, status=result['status'],
                          validation=result['validation'])), flush=True)


def run_jobs(root, jobs, workers, phase):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def launch(job):
        target, seed, start = job
        path = root / 'etas_logs' / f'{phase}__{target}__{start}__s{seed}.log'
        path.parent.mkdir(exist_ok=True)
        with path.open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(root / 'etas_driver.py'), phase,
                '--root', str(root), '--target', target, '--seed', str(seed), '--start', start],
                env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f'COMPLETE {phase} {target} {start} s{seed}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            future.result()


def run(args):
    root = args.root.resolve()
    plan = read(root / 'etas_plan.json')
    verify(root, plan)
    targets = list(plan['targets'])
    write(root / 'etas_execution.json', dict(workers=args.workers, started_unix=time.time()))
    run_jobs(root, [(t, 0, start) for t in targets for start in plan['starts']], args.workers, 'fit')
    selection = {}
    for target in targets:
        candidates = [read(root / 'etas' / target / f'{start}__s0/fit.json') for start in plan['starts']]
        winner = min(candidates, key=lambda r: r['validation'])
        selection[target] = dict(start=winner['start'], validation=winner['validation'], candidates=candidates)
    write(root / 'etas_selection.json', selection)
    run_jobs(root, [(t, s, selection[t]['start']) for t in targets for s in [1, 2]], args.workers, 'fit')
    run_jobs(root, [(t, s, selection[t]['start']) for t in targets for s in [0, 1, 2]], args.workers, 'score')
    rows = [read(root / 'etas' / t / f"{selection[t]['start']}__s{s}/test.json")
            for t in targets for s in [0, 1, 2]]
    summary = {t: dict(mean=statistics.mean(r['test_nll'] for r in rows if r['target'] == t),
                       sd=statistics.stdev(r['test_nll'] for r in rows if r['target'] == t)) for t in targets}
    write(root / 'etas_comparison.json', dict(summary=summary, selection=selection, rows=rows,
          enforce_p_gt_one=False, enforce_subcritical=False))
    write(root / 'etas_complete.json', dict(passed=True, targets=targets))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run', 'fit', 'score'])
    parser.add_argument('--root', type=Path, default=BASE / 'benchmark_convergence_h4096_3seed')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--target')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start', choices=STARTS, default='warm_raw')
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'fit': worker, 'score': worker}[args.phase](args)
