"""Isolate external rotary clocks while preserving real delta-t features and labels."""
import argparse
import copy
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ORIGIN = Path('/root/autodl-tmp/em_eqf')
STORE = Path('/root/autodl-fs/em_eqf_training/mamba_phase_ablation_20261105')
REPORTS = STORE / 'reports'
RUNS = STORE / 'runs'
FROZEN = STORE / 'frozen_project'
PREVIOUS = ORIGIN / 'experiments/revision/mamba_backbones_20261105/reports'
DATA = STORE / 'dataset_snapshot.pt'
ARMS = [('siso_native', False, 'native'), ('siso_index', False, 'index'),
        ('siso_real', False, 'real'), ('siso_surrogate', False, 'surrogate'),
        ('mimo_native', True, 'native'), ('mimo_real', True, 'real')]
SEEDS = [0, 1, 2]
STEPS = 1500
EVERY = 100
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/run')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PhaseClockAdapter:
    """Wrap the existing adapter without changing any of its real-data outputs."""
    def __init__(self, delegate, mode):
        if mode not in {'native', 'index', 'real', 'surrogate'}:
            raise ValueError(mode)
        self.delegate, self.mode = delegate, mode

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __call__(self, batch):
        result = self.delegate(batch)
        clock = result['times']
        if self.mode == 'native':
            result['phase_times'] = torch.zeros_like(clock)
        elif self.mode == 'index':
            # One normalized training mean interval per event; no held-out scale fit.
            result['phase_times'] = torch.arange(1, clock.shape[1] + 1,
                dtype=clock.dtype, device=clock.device)[None].expand_as(clock) * result['input_mask']
        elif self.mode == 'surrogate':
            result['phase_times'] = batch.phase_clock.to(clock) * result['input_mask']
        return result


def surrogate_clock(data, region, split, seed, length=None):
    """Independent renewal clock; samples only from the training interval bank."""
    spec = data['regions'][region]
    bank = np.diff(spec['splits']['train']['times']) / spec['metadata'][2]
    assert np.isfinite(bank).all() and (bank > 0).all()
    count = len(spec['splits'][split]['times']) if length is None else length
    tag = f'phase-clock-v1:{seed}:{region}:{split}'.encode()
    clock_seed = int.from_bytes(hashlib.sha256(tag).digest()[:8], 'little')
    rng = np.random.default_rng(clock_seed)
    return np.cumsum(rng.choice(bank, size=count, replace=True), dtype=np.float64)


def window(source, left, context, scored, metadata, region_id, fake):
    from run_long_multiregion import sequence
    seq = sequence(source, left, context, scored, metadata, region_id)
    right = left + context + scored
    seq.phase_clock = torch.from_numpy((fake[left:right] - fake[left - 1]).copy())
    assert len(seq.phase_clock) == len(seq)
    return seq


def pools(data, split, seed):
    result = {}
    for region, spec in data['regions'].items():
        if split != 'train' and not spec['core']:
            continue
        context, scored = (256, 1024) if split == 'train' else (256, 128)
        starts = (range(257, 257 + spec['train_targets'], scored) if split == 'train'
                  else range(257, len(spec['splits'][split]['times']) - scored + 1, scored))
        fake = surrogate_clock(data, region, split, seed)
        result[region] = [window(spec['splits'][split], start - context, context, scored,
            spec['metadata'], data['vocabulary'][region], fake) for start in starts]
        assert result[region]
    return result


def model_config(is_mimo):
    plan = json.loads((REPORTS / 'base_plan.json').read_text())
    name = 'mamba3_rotary_mimo' if is_mimo else 'mamba3_rotary_siso'
    config = copy.deepcopy(plan['backbones'][name])
    config['mixer_model_config']['use_residual'] = True
    config['mixer_time_parametrization'] = 'legacy'
    assert config['features_input_keys'] == ['mag', 'log_inter_times']
    assert config['mixer_model_config']['attn_layer_idx'] == []
    return config


def create_model(is_mimo, mode, seed, vocabulary):
    from run_multidomain_pretraining import build_model
    model = build_model('stats_film', seed, vocabulary, model_config(is_mimo))
    # Reuse the earlier study's fixed, pre-training common initialization only.
    # Every fit still starts from scratch; no trained weights are loaded here.
    shared = torch.load(REPORTS / f'common_initialization_seed_{seed}.pth',
                        map_location='cpu', weights_only=False)
    state = model.state_dict()
    for name, value in shared.items():
        assert name in state and value.shape == state[name].shape
        state[name] = value.to(state[name].device)
    model.load_state_dict(state, strict=True)
    model.base_model.input_adapter = PhaseClockAdapter(model.base_model.input_adapter, mode)
    torch.manual_seed(seed + 1000)
    return model


def evaluate(model, items):
    from run_multidomain_pretraining import evaluate as original_evaluate
    return original_evaluate(model, items)


def fit(data, train, validation, arm, is_mimo, mode, seed):
    from run_multidomain_pretraining import make_batch
    folder = RUNS / f'{arm}_seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(arm=arm, is_mimo=is_mimo, phase_mode=mode, seed=seed,
        backbone=model_config(is_mimo), dataset_sha256=sha(DATA), max_updates=STEPS,
        checkpoint_selection='validation macro joint NLL', clock_seed_policy='phase-clock-v1')
    done = folder / 'training_complete.json'
    if done.exists():
        saved = json.loads(done.read_text())
        assert saved['config'] == config
        assert sha(folder / 'best_model.pth') == saved['checkpoint_sha256']
        return saved
    save_json(folder / 'config.json', config)
    model = create_model(is_mimo, mode, seed, data['vocabulary'])
    initial_hash = tensor_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng, order = random.Random(seed), hashlib.sha256()
    regions = sorted(train)
    best, history = math.inf, []
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        selected = []
        for _ in range(2):
            region = rng.choice(regions)
            index = rng.randrange(len(train[region]))
            selected.append(train[region][index])
            order.update(f'{region}:{index};'.encode())
        batch = make_batch(selected)
        assert int(batch.nll_event_mask.sum()) == 2048
        loss = model.nll_loss(batch, reduction='per_event')['total'].mean()
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss: {arm}, {seed}, {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError(f'Nonfinite gradient: {arm}, {seed}, {step}')
        lr = 3e-4 * min(1., step / 150) * max(.05, .5 * (1 + math.cos(math.pi * step / STEPS)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if step % EVERY == 0:
            metrics, _ = evaluate(model, validation)
            row = dict(step=step, train_nll=float(loss), validation=metrics,
                       elapsed_seconds=time.time() - started)
            history.append(row)
            if metrics['macro']['total'] < best:
                best = metrics['macro']['total']
                temp = folder / 'best_model.tmp'
                torch.save(dict(model_state_dict=model.state_dict(), config=config,
                    step=step, validation=metrics), temp)
                temp.replace(folder / 'best_model.pth')
            save_json(folder / 'progress.json', dict(config=config, history=history))
            print(json.dumps(dict(run=folder.name, step=step, validation=metrics['macro'],
                seconds=round(time.time() - started, 1))), flush=True)
    best_payload = torch.load(folder / 'best_model.pth', map_location='cpu', weights_only=False)
    result = dict(config=config, initial_state_sha256=initial_hash,
        data_order_sha256=order.hexdigest(), parameters=sum(p.numel() for p in model.parameters()),
        best_step=best_payload['step'], validation=best_payload['validation'],
        checkpoint_sha256=sha(folder / 'best_model.pth'), elapsed_seconds=time.time() - started,
        history=history)
    save_json(done, result)
    print('TRAINING_COMPLETE', folder.name, flush=True)
    del model, optimizer, best_payload
    torch.cuda.empty_cache()
    return result


def preflight(data):
    from run_multidomain_pretraining import make_batch
    train = pools(data, 'train', 0)
    batch = make_batch([train[r][0] for r in sorted(train)[:2]])
    rows, hashes = [], {}
    for arm, mimo, mode in ARMS:
        model = create_model(mimo, mode, 0, data['vocabulary'])
        adapter = model.base_model.input_adapter
        original = adapter.delegate(batch)
        actual = adapter(batch)
        assert original.keys() <= actual.keys()
        assert all(torch.equal(value, actual[name]) for name, value in original.items())
        initial = tensor_hash(model.state_dict())
        if mimo in hashes:
            assert hashes[mimo] == initial
        hashes[mimo] = initial
        losses = model.nll_loss(batch, reduction='per_event')
        losses['total'].mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        rows.append(dict(arm=arm, loss=float(losses['total'].mean()),
            parameters=sum(p.numel() for p in model.parameters()),
            real_features_and_times_unchanged=True, initial_state_sha256=initial))
        print('PREFLIGHT', json.dumps(rows[-1]), flush=True)
        del model
        torch.cuda.empty_cache()
    save_json(REPORTS / 'preflight.json', rows)


def summarize(rows):
    aggregates = []
    for arm, _, _ in ARMS:
        for region in ['italy', 'japan_kyushu', 'new_zealand', 'macro']:
            group = [r for r in rows if r['arm'] == arm and r['region'] == region]
            aggregates.append(dict(arm=arm, region=region, n=len(group),
                parameters=group[0]['parameters'], **{k:dict(mean=statistics.mean(r[k] for r in group),
                sd=statistics.stdev(r[k] for r in group)) for k in ['time', 'mag', 'total']}))
    comparisons = []
    for real, control in [('siso_real', 'siso_native'), ('siso_real', 'siso_index'),
                          ('siso_real', 'siso_surrogate'), ('mimo_real', 'mimo_native')]:
        for region in ['italy', 'japan_kyushu', 'new_zealand', 'macro']:
            values = {}
            for metric in ['time', 'mag', 'total']:
                gains = []
                for seed in SEEDS:
                    baseline = next(r for r in rows if r['arm'] == control and r['seed'] == seed and r['region'] == region)
                    treatment = next(r for r in rows if r['arm'] == real and r['seed'] == seed and r['region'] == region)
                    gains.append(baseline[metric] - treatment[metric])
                values[metric] = dict(per_seed=gains, mean=statistics.mean(gains),
                    sd=statistics.stdev(gains), positive_seeds=sum(g > 0 for g in gains))
            comparisons.append(dict(real=real, control=control, region=region, gains=values))
    return aggregates, comparisons


def finalize(data):
    completed = {}
    for seed in SEEDS:
        for arm, _, _ in ARMS:
            folder = RUNS / f'{arm}_seed_{seed}'
            saved = json.loads((folder / 'training_complete.json').read_text())
            assert sha(folder / 'best_model.pth') == saved['checkpoint_sha256']
            completed[arm, seed] = saved
        for mimo in [False, True]:
            values = [completed[a, seed] for a, m, _ in ARMS if m == mimo]
            assert len({v['initial_state_sha256'] for v in values}) == 1
            assert len({v['parameters'] for v in values}) == 1
        assert len({completed[a, seed]['data_order_sha256'] for a, _, _ in ARMS}) == 1
    # Do not construct test windows until every planned training fit is complete.
    rows, interventions = [], []
    for seed in SEEDS:
        tests = pools(data, 'test', seed)
        for arm, mimo, mode in ARMS:
            folder = RUNS / f'{arm}_seed_{seed}'
            result = completed[arm, seed]
            model = create_model(mimo, mode, seed, data['vocabulary'])
            payload = torch.load(folder / 'best_model.pth', map_location='cuda:0', weights_only=False)
            model.load_state_dict(payload['model_state_dict'], strict=True)
            metrics, windows = evaluate(model, tests)
            save_json(folder / 'metrics.json', dict(**result, test=metrics))
            save_json(folder / 'test_window_nll.json', windows)
            for region in [*metrics['regions'], 'macro']:
                value = metrics['macro'] if region == 'macro' else metrics['regions'][region]
                rows.append(dict(arm=arm, seed=seed, region=region, **value,
                    parameters=result['parameters'], best_step=result['best_step']))
            if mode == 'real':
                for replacement in ['native', 'index', 'surrogate']:
                    model.base_model.input_adapter.mode = replacement
                    perturbed, _ = evaluate(model, tests)
                    interventions.append(dict(arm=arm, seed=seed, phase_replacement=replacement,
                        original=metrics, perturbed=perturbed))
                model.base_model.input_adapter.mode = 'real'
            del model, payload
            torch.cuda.empty_cache()
            print('TEST_COMPLETE', folder.name, json.dumps(metrics['macro']), flush=True)
    with (REPORTS / 'all_results.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates, comparisons = summarize(rows)
    save_json(REPORTS / 'summary.json', dict(completed_fits=len(completed),
        metrics=aggregates, paired_gains=comparisons, initialization_and_data_order_verified=True,
        primary_metric='macro time NLL', checkpoint_selection='validation macro joint NLL',
        units='nats/event; lower is better; positive paired gain favors real phase'))
    save_json(REPORTS / 'test_phase_interventions.json', interventions)
    print('PHASE_ABLATION_COMPLETE', REPORTS / 'summary.json', flush=True)


def source_files(root):
    return sorted([*root.joinpath('src').rglob('*.py'),
                   *root.joinpath('scripts/run').rglob('*.py')])


def verify_snapshot():
    plan = json.loads((REPORTS / 'study_plan.json').read_text())
    assert sha(DATA) == plan['dataset_sha256']
    assert all(sha(FROZEN / name) == value for name, value in plan['source_sha256'].items())
    return plan


def launch():
    import importlib.metadata
    STORE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if (REPORTS / 'study_plan.json').exists():
        raise RuntimeError('Existing frozen study. Resume its worker explicitly; do not replace it.')
    base_plan = json.loads((PREVIOUS / 'study_plan.json').read_text())
    original_data = Path('/root/autodl-fs/em_eqf_training/mamba_backbones_20261105/dataset_snapshot.pt')
    shutil.copyfile(original_data, DATA)
    assert sha(DATA) == base_plan['dataset_sha256']
    save_json(REPORTS / 'base_plan.json', base_plan)
    for seed in SEEDS:
        shutil.copyfile(PREVIOUS / f'common_initialization_seed_{seed}.pth',
                        REPORTS / f'common_initialization_seed_{seed}.pth')
    paths = source_files(ORIGIN)
    for source in paths:
        target = FROZEN / source.relative_to(ORIGIN)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    test = ORIGIN / 'tests/model/test_mamba3_phase_ablation.py'
    target = FROZEN / 'tests/model/test_mamba3_phase_ablation.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test, target)
    checks = []
    for source in paths:
        assert sha(source) == sha(FROZEN / source.relative_to(ORIGIN)), source
        checks.append((str(source.relative_to(ORIGIN)), sha(source)))
    plan = dict(created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        protocol='external_phase_only_v1', arms=[dict(arm=a, is_mimo=m, phase_mode=p) for a,m,p in ARMS],
        seeds=SEEDS, updates=STEPS, validation_every=EVERY, dataset_sha256=sha(DATA),
        split=base_plan['split'], sources=base_plan['sources'], targets=base_plan['targets'],
        input_features=['mag', 'log_inter_times'], input_dim=2, use_residual=True,
        time_head='legacy', shared_initialization='previous pre-training common initialization; no trained weights',
        training_context=256, training_scored=1024, batch_size=2, evaluation_context=256,
        evaluation_scored=128, learning_rate=3e-4, optimizer='AdamW', weight_decay=.01,
        gradient_clip=1., scheduler='10% warmup then cosine floor 5%',
        checkpoint_selection='minimum validation macro joint NLL', primary_metric='test macro time NLL',
        secondary_metrics=['joint NLL', 'magnitude NLL', 'per-region NLL'],
        surrogate='independent renewal clock sampled from regional train-only interval bank / calibration tau',
        surrogate_replicates='one deterministic independent clock per model seed, region and split',
        index='one normalized mean interval per event, i=1,2,...',
        fixed=['real delta-t features', 'labels', 'loss times', 'native learned DT', 'heads', 'width', 'initialization', 'data order'],
        test_access='after all 18 fits complete; predefined test-time interventions, no test tuning',
        source_sha256=dict(checks))
    save_json(REPORTS / 'study_plan.json', plan)
    with tarfile.open(REPORTS / 'code_snapshot.tar.gz', 'w:gz') as archive:
        for name in plan['source_sha256']:
            archive.add(FROZEN / name, arcname=name)
        archive.add(target, arcname='tests/model/test_mamba3_phase_ablation.py')
    save_json(REPORTS / 'environment.json', dict(python=sys.executable, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        packages={name:importlib.metadata.version(name) for name in ['mamba-ssm','triton','tilelang']}))
    link = ORIGIN / 'experiments/revision/mamba_phase_ablation_20261105'
    if not link.exists():
        link.symlink_to(STORE, target_is_directory=True)
    log = (REPORTS / 'training.log').open('ab', buffering=0)
    env = dict(os.environ, OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', str(FROZEN / 'scripts/run/run_mamba_phase_ablation.py'), '--worker']
    process = subprocess.Popen(command, cwd=FROZEN, env=env, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
    (REPORTS / 'process.pid').write_text(str(process.pid))
    print('LAUNCHED', process.pid, 'LOG', REPORTS / 'training.log', flush=True)


def worker():
    torch.set_num_threads(1)
    RUNS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_snapshot()
        data = torch.load(DATA, map_location='cpu', weights_only=False)
        preflight(data)
        for seed in SEEDS:
            train, validation = pools(data, 'train', seed), pools(data, 'val', seed)
            for arm, mimo, mode in ARMS:
                fit(data, train, validation, arm, mimo, mode, seed)
        verify_snapshot()
        finalize(data)
        verify_snapshot()
        save_json(REPORTS / 'completed.json', dict(completed_fits=18, snapshots_verified=True,
            completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))


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
