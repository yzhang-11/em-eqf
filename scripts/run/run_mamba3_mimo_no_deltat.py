"""Retrain the real-time Mamba3 MIMO arm without direct delta-t information."""
import argparse
import csv
import datetime
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[2]
ORIGIN = Path('/root/autodl-tmp/em_eqf')
BASE = Path('/root/autodl-fs/em_eqf_training/mamba_phase_ablation_20261105')
STORE = Path('/root/autodl-fs/em_eqf_training/mamba3_mimo_no_deltat_20261105')
REPORTS = STORE / 'shared/reports'
RUNS = STORE / 'runs'
FROZEN = STORE / 'frozen_project'
DATA = STORE / 'dataset_snapshot.pt'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/run')]
SEEDS = [0, 1, 2]
ARM = 'mimo_real_no_deltat'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)


class NoDeltaTAdapter:
    """Keep real clocks and labels; locate the interval column by feature name."""
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, key):
        return getattr(self.delegate, key)

    def __call__(self, batch):
        result = self.delegate(batch)
        features = result['features']
        names = list(self.features_input_keys)
        if features.shape[-1] != 2 or len(names) != 2 or set(names) != {'mag', 'log_inter_times'}:
            raise ValueError('Expected exactly the magnitude and log interval features')
        masked = features.clone()
        masked[..., names.index('log_inter_times')] = 0
        result['features'] = masked
        return result


def configure_phase_runner():
    import run_mamba_phase_ablation as phase
    phase.STORE, phase.REPORTS, phase.RUNS = STORE, REPORTS, RUNS
    phase.FROZEN, phase.DATA = FROZEN, DATA
    original_config = phase.model_config
    original_create = phase.create_model

    def config(is_mimo):
        assert is_mimo
        result = original_config(is_mimo)
        result['direct_delta_t_input'] = 'masked_zero_after_standardization'
        return result

    def create(is_mimo, mode, seed, vocabulary):
        assert is_mimo and mode == 'real'
        model = original_create(is_mimo, mode, seed, vocabulary)
        model.base_model.input_adapter = NoDeltaTAdapter(model.base_model.input_adapter)
        return model

    phase.model_config, phase.create_model = config, create
    return phase


def verify_snapshot():
    plan = json.loads((REPORTS / 'study_plan.json').read_text())
    assert sha(DATA) == plan['dataset_sha256']
    assert all(sha(FROZEN / name) == h for name, h in plan['source_sha256'].items())
    native = Path(glob.glob('/root/autodl-tmp/venvs/em_eqf-cu128/lib/python*/site-packages/mamba_ssm')[0])
    hashes = json.loads((REPORTS / 'native_kernels_sha256.json').read_text())
    assert all(sha(native / name) == h for name, h in hashes.items())
    return plan


def preflight(phase, data):
    from run_multidomain_pretraining import make_batch
    train = phase.pools(data, 'train', 0)
    batch = make_batch([train[r][0] for r in sorted(train)[:2]])
    model = phase.create_model(True, 'real', 0, data['vocabulary'])
    adapter = model.base_model.input_adapter
    original = adapter.delegate(batch)
    actual = adapter(batch)
    names = list(adapter.features_input_keys)
    dt_index, mag_index = names.index('log_inter_times'), names.index('mag')
    assert torch.equal(original['features'][..., mag_index], actual['features'][..., mag_index])
    assert torch.count_nonzero(actual['features'][..., dt_index]) == 0
    assert all(torch.equal(v, actual[k]) for k,v in original.items() if k != 'features')
    baseline = json.loads((REPORTS / 'baseline_seed_0.json').read_text())
    assert phase.tensor_hash(model.state_dict()) == baseline['initial_state_sha256']
    loss = model.nll_loss(batch, reduction='per_event')['total'].mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    first = next(m for m in model.base_model.encoder.input_proj.modules()
                 if isinstance(m, torch.nn.Linear) and m.in_features == 2)
    assert torch.count_nonzero(first.weight.grad[:, dt_index]) == 0
    checks = dict(magnitude_features_unchanged=True, direct_delta_t_channel_zero=True,
        real_time_and_other_adapter_outputs_unchanged=True,
        initial_weights_identical_to_baseline=True, finite_forward_backward=True,
        masked_input_projection_gradient_zero=True, actual_feature_order=names,
        masked_feature_index=dt_index, parameters=sum(p.numel() for p in model.parameters()))
    save_json(REPORTS / 'preflight.json', checks)
    print('PREFLIGHT_PASSED', json.dumps(checks), flush=True)
    del model
    torch.cuda.empty_cache()


def finalize(phase, data):
    completed = {seed:json.loads((RUNS / f'{ARM}_seed_{seed}' / 'training_complete.json').read_text())
                 for seed in SEEDS}
    # Evaluate only after all three new fits have completed.
    rows = []
    for seed, result in completed.items():
        baseline = json.loads((REPORTS / f'baseline_seed_{seed}.json').read_text())
        assert result['initial_state_sha256'] == baseline['initial_state_sha256']
        assert result['data_order_sha256'] == baseline['data_order_sha256']
        assert result['parameters'] == baseline['parameters'] == 158609
        folder = RUNS / f'{ARM}_seed_{seed}'
        assert sha(folder / 'best_model.pth') == result['checkpoint_sha256']
        model = phase.create_model(True, 'real', seed, data['vocabulary'])
        state = torch.load(folder / 'best_model.pth', map_location='cuda:0', weights_only=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        metrics, windows = phase.evaluate(model, phase.pools(data, 'test', seed))
        save_json(folder / 'metrics.json', dict(**result, test=metrics))
        save_json(folder / 'test_window_nll.json', windows)
        for name, values in [('with_delta_t', baseline['test']), ('without_delta_t', metrics)]:
            for region in [*values['regions'], 'macro']:
                value = values['macro'] if region == 'macro' else values['regions'][region]
                rows.append(dict(variant=name, seed=seed, region=region, **value,
                    best_step=baseline['best_step'] if name == 'with_delta_t' else result['best_step'],
                    parameters=result['parameters']))
        print('TEST_COMPLETE', seed, json.dumps(metrics['macro']), flush=True)
        del model, state
        torch.cuda.empty_cache()
    with (REPORTS / 'all_results.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates, differences = [], []
    for variant in ['with_delta_t', 'without_delta_t']:
        for region in ['italy', 'japan_kyushu', 'new_zealand', 'macro']:
            group = [r for r in rows if r['variant'] == variant and r['region'] == region]
            aggregates.append(dict(variant=variant, region=region, n=3,
                **{k:dict(mean=statistics.mean(r[k] for r in group),
                          sd=statistics.stdev(r[k] for r in group)) for k in ['time','mag','total']}))
    for region in ['italy', 'japan_kyushu', 'new_zealand', 'macro']:
        values = {}
        for key in ['time','mag','total']:
            delta = [next(r[key] for r in rows if r['variant'] == 'without_delta_t' and r['seed'] == seed and r['region'] == region)
                     - next(r[key] for r in rows if r['variant'] == 'with_delta_t' and r['seed'] == seed and r['region'] == region)
                     for seed in SEEDS]
            values[key] = dict(mean=statistics.mean(delta), sd=statistics.stdev(delta),
                               per_seed=delta, worse_seeds=sum(x > 0 for x in delta))
        differences.append(dict(region=region, without_minus_with=values))
    save_json(REPORTS / 'summary.json', dict(completed_new_fits=3, metrics=aggregates,
        differences=differences, initialization_and_sampling_match_baseline=True,
        real_time_phase_retained=True, parameters=158609,
        delta_t_removal='constant zero standardized feature column; network shape unchanged',
        units='nats/event, lower is better; positive difference means removal is worse'))


def worker():
    assert ROOT.resolve() == FROZEN.resolve()
    torch.set_num_threads(1)
    RUNS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_snapshot()
        phase = configure_phase_runner()
        data = torch.load(DATA, map_location='cpu', weights_only=False)
        preflight(phase, data)
        for seed in SEEDS:
            phase.fit(data, phase.pools(data, 'train', seed), phase.pools(data, 'val', seed),
                      ARM, True, 'real', seed)
        verify_snapshot()
        finalize(phase, data)
        verify_snapshot()
        save_json(REPORTS / 'completed.json', dict(completed_new_fits=3, snapshots_verified=True,
            completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
        print('NO_DELTAT_COMPLETE', REPORTS / 'summary.json', flush=True)


def launch():
    STORE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if (REPORTS / 'study_plan.json').exists():
        raise RuntimeError('Existing frozen study; resume its frozen worker instead of replacing files')
    previous = json.loads((BASE / 'reports/study_plan.json').read_text())
    sources = {}
    for name, expected in previous['source_sha256'].items():
        source = BASE / 'frozen_project' / name
        assert sha(source) == expected
        target = FROZEN / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[name] = expected
    for name in ['scripts/run/run_mamba3_mimo_no_deltat.py', 'tests/model/test_mamba3_no_delta_input.py']:
        target = FROZEN / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ORIGIN / name, target)
        sources[name] = sha(target)
    shutil.copyfile(BASE / 'dataset_snapshot.pt', DATA)
    for name in ['base_plan.json', 'environment.json', 'native_kernels_sha256.json']:
        shutil.copyfile(BASE / 'shared/reports' / name, REPORTS / name)
    for seed in SEEDS:
        shutil.copyfile(BASE / 'shared/reports' / f'common_initialization_seed_{seed}.pth',
                        REPORTS / f'common_initialization_seed_{seed}.pth')
        shutil.copyfile(BASE / 'runs' / f'mimo_real_seed_{seed}' / 'metrics.json',
                        REPORTS / f'baseline_seed_{seed}.json')
    plan = dict(protocol='mimo_real_remove_direct_delta_t_v1', seeds=SEEDS, updates=1500,
        baseline='previous frozen MIMO native + real event-time phase, with direct log(delta-t)',
        ablation='zero standardized log_inter_times channel for all training/validation/test inputs',
        input_shape=2, informative_input_features=['mag'], parameters=158609,
        phase='native learned phase + real event time / calibration tau',
        fixed=['all initial weights','sampled data order','architecture','real time phase','targets',
               'time/magnitude heads','residual enabled','legacy time head','learning rate and schedule'],
        checkpoint_selection='validation macro joint NLL', data_split=previous['split'],
        train_context=256, train_scored=1024, batch_size=2, evaluation_context=256, evaluation_scored=128,
        optimizer='AdamW', learning_rate=3e-4, weight_decay=.01, gradient_clip=1.,
        scheduler='10% warmup then cosine floor 5%', validation_every=100,
        test_access='only after all three new training fits complete',
        dataset_sha256=previous['dataset_sha256'], source_sha256=sources,
        created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    save_json(REPORTS / 'study_plan.json', plan)
    verify_snapshot()
    with tarfile.open(REPORTS / 'code_snapshot.tar.gz', 'w:gz') as output:
        for name in sources:
            output.add(FROZEN / name, arcname=name)
    link = ORIGIN / 'experiments/revision/mamba3_mimo_no_deltat_20261105'
    if not link.exists():
        link.symlink_to(STORE, target_is_directory=True)
    log = (REPORTS / 'training.log').open('ab', buffering=0)
    process = subprocess.Popen([sys.executable, '-u', str(FROZEN / 'scripts/run/run_mamba3_mimo_no_deltat.py'), '--worker'],
        cwd=FROZEN, env=dict(os.environ, OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1'),
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (REPORTS / 'process.pid').write_text(str(process.pid))
    print('LAUNCHED', process.pid, 'LOG', REPORTS / 'training.log', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.run:
        launch()
    elif args.worker:
        worker()
    else:
        parser.error('Choose --run or --worker')
