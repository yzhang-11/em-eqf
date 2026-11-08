"""Run paired I/J-NMRP baselines against an audited RECAST benchmark study."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
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
MODELS = ('i-nmrp', 'j-nmrp')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def runtime(project):
    sys.path[:0] = [str(project), str(project / 'scripts/run')]
    import torch
    location = importlib.util.spec_from_file_location('nmrp_native_fitter', project.parent / 'native_fitter.py')
    native = importlib.util.module_from_spec(location)
    location.loader.exec_module(native)
    from run_expanded_multiregion import calendar_bound, digest, spans
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    return torch, native, calendar_bound, digest, spans


def prepare(args):
    reference = args.reference.resolve()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'project').symlink_to(reference / 'project', target_is_directory=True)
    # One fitter overlay avoids copying hundreds of shared snapshot files.
    fitter = Path(__file__).resolve().parent / 'run_unified_tpp.py'
    text = fitter.read_text()
    origin = 'ROOT = Path(__file__).resolve().parents[2]'
    assert text.count(origin) == 1
    text = text.replace(origin, f'ROOT = Path({str(reference / "project")!r})')
    (root / 'native_fitter.py').write_text(text)
    shutil.copy2(Path(__file__), root / 'driver.py')
    torch, native, _, _, _ = runtime(root / 'project')
    ref = read(reference / 'plan.json')
    comparison = read(reference / 'gr/comparison.json')
    critical = ['scripts/run/run_unified_tpp.py', 'scripts/run/run_expanded_multiregion.py',
                'src/models/tpp/nmrp.py', 'src/models/tpp/tpp_model.py',
                'src/data/unified_tpp.py', 'src/data/batch.py', 'src/data/sequence.py',
                'src/train/scheduler.py']
    cases = []
    for target in ref['cohorts']:
        data_dir = reference / 'data' / target
        manifest = read(data_dir / 'dataset.json')
        assert sha(data_dir / 'development.pt') == manifest['artifacts']['development.pt']
        development = native.load_catalogs(data_dir)
        region = development['regions'][target]
        for seed in (0, 1, 2):
            reference_row, = [r for r in comparison['rows']
                              if (r['target'], r['seed'], r['mode']) == (target, seed, 'local')]
            for model in MODELS:
                spec = dict(model=model, target=target, seed=seed, lr=.001, steps=800,
                            validate_every=100, history=4096, train_events=2048,
                            nmrp_history=50, objective='time_nll',
                            manifest=sha(data_dir / 'dataset.json'),
                            region_id=development['vocabulary'][target],
                            draw_sha256=reference_row['draw_sha256'])
                cases.append(dict(name=f'{target}__{model}__s{seed}', data=str(data_dir),
                                  spec=spec, expected_test_events=reference_row['events']))
    plan = dict(reference=str(reference), cases=cases, models=MODELS, seeds=[0, 1, 2],
                protocol='full_sequence_calendar', driver_sha256=sha(root / 'driver.py'),
                fitter_sha256=sha(root / 'native_fitter.py'),
                source_files={f: sha(root / 'project' / f) for f in critical},
                reference_files={str(p): sha(p) for p in [reference / 'gr/comparison.json',
                    reference / 'final_comparison.json', reference / 'cohort_audit.json']})
    write(root / 'plan.json', plan)
    print(json.dumps(dict(prepared=str(root), fits=len(cases))), flush=True)


def paired_draws(region, spec, helpers):
    calendar_bound, digest, spans = helpers
    target = spec['target']
    pool = spans(region['train_start'], region['train_end'], spec['train_events'])
    rng = random.Random(spec['seed'] * 100003 + spec['region_id'] * 1009)
    draws = rng.choices(pool, weights=[count for _, count in pool], k=spec['steps'])
    assert digest({target: draws}) == spec['draw_sha256'], 'RECAST draw pairing failed'
    convert = lambda w: (calendar_bound(region, w[0]), calendar_bound(region, sum(w)), w[1])
    return list(map(convert, pool)), list(map(convert, draws))


def verify(root, plan):
    assert sha(Path(__file__)) == plan['driver_sha256'], 'Driver changed'
    assert sha(root / 'native_fitter.py') == plan['fitter_sha256'], 'Fitter changed'
    for name, value in plan['source_files'].items():
        assert sha(root / 'project' / name) == value, f'Source changed: {name}'
    for name, value in plan['reference_files'].items():
        assert sha(Path(name)) == value, f'Reference changed: {name}'


def worker(root, name):
    plan = read(root / 'plan.json')
    verify(root, plan)
    torch, native, *helpers = runtime(root / 'project')
    case, = [c for c in plan['cases'] if c['name'] == name]
    spec = case['spec']
    folder = root / 'runs' / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        result = read(folder / 'result.json')
        assert sha(folder / 'best.pt') == result['checkpoint_sha256']
        assert result['spec'] == spec
        return
    data_dir = Path(case['data'])
    manifest = read(data_dir / 'dataset.json')
    assert sha(data_dir / 'dataset.json') == spec['manifest']
    assert sha(data_dir / 'development.pt') == manifest['artifacts']['development.pt']
    data = native.load_catalogs(data_dir)
    region = data['regions'][spec['target']]
    paired_draws(region, spec, helpers)
    device = torch.device('cuda')
    model = native.make_model(spec, data, region, device)
    parameters = sum(p.numel() for p in model.parameters())
    def full_evaluate(*arguments):
        # Keep training parallel; serialize the large whole-catalog evaluations.
        torch.cuda.empty_cache()
        with (root / 'evaluation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = native.evaluate(*arguments)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return value
    started = time.time()
    if not (folder / 'complete.json').exists():
        native.fit(model, spec, region, spec['region_id'], folder, device,
                   draw_function=lambda r, s: paired_draws(r, s, helpers),
                   evaluation_function=full_evaluate)
        write(folder / 'complete.json', dict(spec=spec, checkpoint_sha256=sha(folder / 'best.pt')))
    complete = read(folder / 'complete.json')
    assert complete['spec'] == spec and complete['checkpoint_sha256'] == sha(folder / 'best.pt')
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    assert checkpoint['spec'] == spec
    model.load_state_dict(checkpoint['state'], strict=True)
    dates = region['time_bounds_ns']
    validation = full_evaluate(model, spec, region, spec['region_id'],
                                 dates['train_end'], dates['val_end'], device)
    assert abs(validation['time_nll_per_event'] - checkpoint['validation']['time_nll_per_event']) < 1e-8
    assert sha(data_dir / 'test.pt') == manifest['artifacts']['test.pt']
    testing = native.load_catalogs(data_dir, test=True)['regions'][spec['target']]
    dates = testing['time_bounds_ns']
    score = full_evaluate(model, spec, testing, spec['region_id'], dates['val_end'], dates['test_end'], device)
    assert score['events'] == case['expected_test_events']
    write(folder / 'test.json', score)
    result = dict(target=spec['target'], model=spec['model'], seed=spec['seed'], spec=spec,
                  validation=validation, test=score, best_step=checkpoint['step'],
                  parameters=parameters, seconds=time.time() - started,
                  checkpoint=str(folder / 'best.pt'), checkpoint_sha256=sha(folder / 'best.pt'),
                  peak_memory_bytes=torch.cuda.max_memory_allocated())
    write(folder / 'result.json', result)
    print(json.dumps(result), flush=True)


def smoke(root):
    plan = read(root / 'plan.json')
    verify(root, plan)
    torch, native, *helpers = runtime(root / 'project')
    rows = []
    for name in MODELS:
        case = next(c for c in plan['cases'] if c['spec']['target'] == 'WHITE_06' and c['spec']['model'] == name)
        spec = case['spec']
        data = native.load_catalogs(Path(case['data']))
        region = data['regions'][spec['target']]
        pool, draws = paired_draws(region, spec, helpers)
        model = native.make_model(spec, data, region, torch.device('cuda'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=.01)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.time()
        for left, right, _ in draws[:20]:
            batch = native.calendar_batch(region, left, right, history=4096).to('cuda')
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model.nll_loss(batch, reduction='per_event')
            loss.mean().backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(loss).all() and torch.isfinite(norm)
            optimizer.step()
            model.project_weights()
        torch.cuda.synchronize()
        step_seconds = (time.time() - started) / 20
        probe = root / 'smoke_runs' / name
        probe.mkdir(parents=True, exist_ok=True)
        native.fit(model, dict(spec, steps=10, validate_every=10), region,
                   spec['region_id'], probe, torch.device('cuda'),
                   draw_function=lambda r, s: (pool, draws[:10]))
        dates = region['time_bounds_ns']
        score = native.evaluate(model, spec, region, spec['region_id'], dates['train_end'], dates['val_end'], torch.device('cuda'))
        rows.append(dict(model=name, seconds_per_step=step_seconds, validation=score,
                         peak_memory_bytes=torch.cuda.max_memory_allocated(), paired_draws=True,
                         native_fit_interface=True))
        del model, optimizer, batch, loss
        torch.cuda.empty_cache()
    write(root / 'smoke.json', dict(passed=True, torch_version=torch.__version__,
                                  gpu=torch.cuda.get_device_name(), rows=rows))
    print(json.dumps(rows), flush=True)


def aggregate(root):
    plan = read(root / 'plan.json')
    reference = Path(plan['reference'])
    combined = read(reference / 'final_comparison.json')
    rows = [read(root / 'runs' / c['name'] / 'result.json') for c in plan['cases']]
    summary = {}
    for target, ref in combined['summary'].items():
        table = {f'recast_{m}': v for m, v in ref['gr'].items()}
        table['etas'] = ref['validation_selected_etas']
        for model in MODELS:
            selected = [r for r in rows if r['target'] == target and r['model'] == model]
            assert {r['seed'] for r in selected} == {0, 1, 2}
            values = [r['test']['time_nll_per_event'] for r in selected]
            table[model] = dict(mean=statistics.mean(values), sd=statistics.stdev(values))
        summary[target] = table
    value = dict(metric='per-event physical-day time NLL; lower is better',
                 protocol='full_sequence_calendar', history=4096, train_events=2048,
                 nmrp_history=50, steps=800, nmrp_lr=.001, seeds=[0, 1, 2],
                 paired_recast_target_draws=True, summary=summary, nmrp_runs=rows,
                 reference=str(reference / 'final_comparison.json'))
    write(root / 'comparison.json', value)
    write(root / 'complete.json', dict(passed=True, fits=len(rows), comparison_sha256=sha(root / 'comparison.json')))
    print(json.dumps(summary), flush=True)


def run(args):
    root = args.root.resolve()
    plan = read(root / 'plan.json')
    verify(root, plan)
    assert read(root / 'smoke.json')['passed']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    write(root / 'execution.json', dict(workers=args.workers, cases=len(plan['cases']),
                                     cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                                     started_unix=time.time()))
    def launch(case):
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        with (logs / (case['name'] + '.log')).open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(root / 'driver.py'), 'worker',
                            '--root', str(root), '--name', case['name']],
                           stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
        print('COMPLETE', case['name'], flush=True)
    # Interleave datasets/models across seeds to avoid a long final dataset tail.
    cases = sorted(plan['cases'], key=lambda c: (c['spec']['seed'], c['spec']['target'], c['spec']['model']))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch, c) for c in cases]
        for future in as_completed(futures):
            future.result()
    aggregate(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'smoke', 'run', 'worker', 'aggregate'])
    parser.add_argument('--root', type=Path, default=BASE / 'benchmark_nmrp_h4096_3seed')
    parser.add_argument('--reference', type=Path, default=BASE / 'cross_recast_benchmarks_h4096_3seed')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args)
    elif args.phase == 'run':
        run(args)
    else:
        {'smoke': smoke, 'aggregate': aggregate,
         'worker': lambda r: worker(r, args.name)}[args.phase](args.root.resolve())
