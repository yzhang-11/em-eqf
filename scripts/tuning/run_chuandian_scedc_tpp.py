"""Past-only mixed-source TPP transfer and matched local/ETAS controls."""
import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
TAG = 'chuandian_scedc_tpp_20261105'
BASE = ROOT / 'experiments/revision/multiregion_pretraining'
STORE = Path('/root/autodl-fs/em_eqf_training') / TAG
REPORTS, CONFIGS, RUNS = [BASE / key / TAG for key in ['shared/reports', 'configs', 'runs']]
DATA = STORE / 'targets.pt'
REFERENCE = Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105/frozen_project')
sys.path.insert(0, str(REFERENCE / 'scripts/tuning'))
import mamba3_transfer_common as c

TARGETS = ['chuandian', 'scedc']
FAMILIES = ['mamba2_legacy', 'mamba3_oracle_residual']
SEEDS = [0, 1, 2]
LRS = [3e-5, 1e-4, 3e-4]
STEPS = 800


def read(path):
    return json.loads(Path(path).read_text())


def source_data(target):
    return STORE / f'sources_before_{target}_training_end.pt'


def source_run(target, family, seed):
    return RUNS / f'source_{target}_{family}_seed_{seed}'


def setup():
    c.bootstrap()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True


def build(family, seed, vocabulary):
    from run_multidomain_pretraining import build_model
    cfg = copy.deepcopy(read(c.CONFIGS / 'study_plan.json')['backbones'][family])
    model = build_model('stats_film', seed, vocabulary, cfg)
    assert model.region_embedding is None
    assert model.inter_time_decoder.parametrization == cfg['mixer_time_parametrization']
    if cfg['mixer_time_parametrization'] == 'oracle':
        with torch.no_grad():
            model.hypernet_time.bias[:model.num_components].zero_()
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    return model, cfg


def initialize():
    for p in [STORE, REPORTS, CONFIGS, STORE / 'runs']:
        p.mkdir(parents=True, exist_ok=True)
    if not RUNS.exists():
        RUNS.symlink_to(STORE / 'runs', target_is_directory=True)
    path = CONFIGS / 'study_plan.json'
    if path.exists():
        plan = read(path)
        assert c.sha(STORE / 'frozen_runner.py') == plan['runner_sha256']
        return plan
    shutil.copy2(Path(__file__), STORE / 'frozen_runner.py')
    etas_path = ROOT / 'scripts/tuning/fit_region_etas.py'
    shutil.copy2(etas_path, STORE / 'frozen_etas.py')
    setup()
    target_data = torch.load(DATA, map_location='cpu', weights_only=False)
    configurations, counts = {}, {}
    for family in FAMILIES:
        model, cfg = build(family, 0, target_data['vocabulary'])
        configurations[family] = cfg
        counts[family] = sum(p.numel() for p in model.parameters())
        del model
    torch.cuda.empty_cache()
    plan = dict(protocol=TAG, targets=TARGETS, families=FAMILIES, seeds=SEEDS,
        data_sha256=c.sha(DATA), source_data={t: dict(path=str(source_data(t)), sha256=c.sha(source_data(t))) for t in TARGETS},
        runner_sha256=c.sha(STORE / 'frozen_runner.py'), etas_sha256=c.sha(STORE / 'frozen_etas.py'),
        backbones=configurations, parameters=counts, method='stats_film', layers=2, width=64,
        source_regions=8, source_updates=6400, target_updates=800, target_learning_rates=LRS,
        source_learning_rate=3e-4, optimizer='AdamW', weight_decay=.01, gradient_clip=1.,
        schedule='10% warmup; cosine with5% floor; all trainable parameters share the selected learning rate.',
        sampling='Balanced shuffled source cycles; target blocks sampled proportional to scored events; paired target draws across families/modes.',
        source_selection='Shared8-region validation macro joint NLL; source data and validation end before target training cutoff, with2-day buffer.',
        target_selection='Local and full-finetune each get identical3-rate seed0 grids. Select joint validation NLL, fix rate for seeds1/2.',
        validation_every=100, source_validation_every=800,
        local_long_guard='Before test: if selected seed0 best_step=800 and joint validation improves>0.002 between600 and800, run separate1600-step fits for all3seeds. Same rule for local/fine;800 remains primary.',
        context_events=1024, target_events_per_update=2048, evaluation_targets_per_window=128,
        tasks='Conditional event-time density per day and0.1-bin magnitude mass; report time/mark/joint NLL nats/event; no spatial likelihood.',
        zero_shot='No target weight updates; first256 target training events calibrate normalization/b_ref. Target context is observed causally.',
        etas='Both classic and subcritical temporal ETAS,2 training-likelihood restarts each; family selected on validation, matched1024-history primary and full-history secondary.',
        etas_environment='/root/autodl-tmp/venvs/em_eqf-etas/bin/python',
        test_access='No new test scoring until source, target, budget guards and ETAS fits finish.',
        magnitude='Review-frozen Mc: chuandian3.0,SCEDC2.5; original date splits; lower-edge0.1bins; upper truncation10.',
        limitations=['Historical test periods, exploratory follow-up.', 'Source pretraining adds compute beyond target updates; no claim of equal total compute.',
                    'First256 target events are unsupervised calibration; calibrated zero-shot is not target-data-free.', 'Fixed Mc does not certify post-mainshock completeness.'])
    c.save(path, plan)
    c.save(REPORTS / 'plan.json', plan)
    c.save(REPORTS / 'runtime_environment.json', dict(python=sys.executable, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), precision='high', tf32=True, source_model_code=str(REFERENCE)))
    return plan


def draws_for(data, seed, regions, updates):
    draws = {}
    for region in regions:
        rng = random.Random(seed * 100003 + data['vocabulary'][region] * 1009)
        draws[region] = rng.choices(range(len(data['regions'][region]['block_starts'])),
                                   weights=data['regions'][region]['block_targets'], k=updates)
    if len(regions) == 1:
        order = regions * updates
    else:
        rng = random.Random(seed + 31337)
        order = []
        for _ in range(updates // len(regions)):
            cycle = list(regions)
            rng.shuffle(cycle)
            order.extend(cycle)
    return draws, order


def fit(target, family, seed, mode, lr, updates=800):
    source = mode == 'source'
    folder = source_run(target, family, seed) if source else RUNS / f'{target}_{family}_{mode}_lr_{lr:g}_steps_{updates}_seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'training_complete.json').exists():
        result = read(folder / 'training_complete.json')
        assert c.sha(folder / 'best_model.pth') == result['checkpoint_sha256']
        return result
    setup()
    path = source_data(target) if source else DATA
    data = torch.load(path, map_location='cpu', weights_only=False)
    regions = sorted(data['regions']) if source else [target]
    c._TRAIN_CACHE.clear()
    model, cfg = build(family, seed, data['vocabulary'])
    initial = c.state_hash(model)
    checkpoint = None
    if mode == 'fine':
        checkpoint = source_run(target, family, seed) / 'best_model.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
    loaded = c.state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    draws, order = draws_for(data, seed, regions, updates)
    offsets = dict.fromkeys(regions, 0)
    exposure = dict.fromkeys(regions, 0)
    validation = c.evaluation_pools(data, 'val', 1024)
    validation = {r: validation[r] for r in regions}
    history, best, started = [], math.inf, time.time()
    for step, region in enumerate(order, 1):
        block = draws[region][offsets[region]]
        offsets[region] += 1
        batch = c.training_batch(data, region, block, 'long')
        n = batch.nll_event_mask.sum(-1)
        exposure[region] += int(n.sum())
        model.train()
        losses = model.nll_loss(batch, reduction='per_event')
        loss = (losses['total'] * n).sum() / n.sum()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm)
        factor = min(1., step / max(1, updates // 10)) * max(.05, .5 * (1 + math.cos(math.pi * step / updates)))
        for group in optimizer.param_groups:
            group['lr'] = lr * factor
        optimizer.step()
        every = 800 if source else 100
        if step % every == 0 or step == updates:
            value, _ = c.evaluate(model, validation)
            row = dict(step=step, training_nll=float(loss.detach()), validation=value, seconds=time.time()-started)
            history.append(row)
            if value['macro']['total'] < best:
                best = value['macro']['total']
                torch.save(dict(model_state_dict=model.state_dict(), backbone=cfg, step=step, validation=value), folder / 'best_model.pth')
            c.save(folder / 'progress.json', dict(history=history))
            print('TPP_PROGRESS', folder.name, step, value['macro'], round(time.time()-started, 1), flush=True)
    state = torch.load(folder / 'best_model.pth', map_location='cpu', weights_only=False)
    result = dict(target=target, family=family, seed=seed, mode=mode, lr=lr, updates=updates, backbone=cfg,
        dataset_sha256=c.sha(path), checkpoint=str(folder / 'best_model.pth'), checkpoint_sha256=c.sha(folder / 'best_model.pth'),
        source_checkpoint=str(checkpoint) if checkpoint else None, best_step=state['step'], validation=state['validation'],
        initial_random_sha256=initial, loaded_sha256=loaded, history=history, seconds=time.time()-started,
        parameters=sum(p.numel() for p in model.parameters()), exposures=exposure,
        draw_sha256={r: hashlib.sha256(json.dumps(v[:offsets[r]]).encode()).hexdigest() for r,v in draws.items()}, test_used=False)
    c.save(folder / 'training_complete.json', result)
    del model, optimizer
    c._TRAIN_CACHE.clear()
    torch.cuda.empty_cache()
    return result


def target_worker(target, family, seed):
    for mode in ['local', 'fine']:
        selection_path = CONFIGS / f'lr_{target}_{family}_{mode}.json'
        if seed == 0:
            trials = [fit(target, family, seed, mode, lr) for lr in LRS]
            selected = min(trials, key=lambda r: r['validation']['macro']['total'])
            c.save(selection_path, dict(lr=selected['lr'], test_used=False,
                validation={str(r['lr']): r['validation']['macro'] for r in trials}))
        else:
            fit(target, family, seed, mode, read(selection_path)['lr'])


def etas_fit():
    spec = importlib.util.spec_from_file_location('_chuandian_scedc_etas', STORE / 'frozen_etas.py')
    etas = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = etas
    spec.loader.exec_module(etas)
    etas.DATA, etas.OUT = DATA, REPORTS / 'etas'
    etas.fit_all()


def preflight():
    setup()
    data = torch.load(DATA, map_location='cpu', weights_only=False)
    checks = []
    for family, target in itertools.product(FAMILIES, TARGETS):
        c._TRAIN_CACHE.clear()
        model, _ = build(family, 0, data['vocabulary'])
        batch = c.training_batch(data, target, 0, 'long')
        loss = model.nll_loss(batch, reduction='per_event')['total'].mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert int(batch.nll_event_mask.sum()) == data['regions'][target]['block_targets'][0]
        checks.append(dict(family=family, target=target, finite_loss=float(loss.detach()), finite_gradient=True,
                           scored_events=int(batch.nll_event_mask.sum()), parameters=sum(p.numel() for p in model.parameters())))
        del model
        torch.cuda.empty_cache()
    c._TRAIN_CACHE.clear()
    c.save(REPORTS / 'preflight.json', dict(passed=True, checks=checks))
    print('TPP_PREFLIGHT_COMPLETE', flush=True)


def jobs_run(jobs, workers):
    active = []
    while jobs or active:
        while jobs and len(active) < workers:
            phase, target, family, seed = jobs.pop(0)
            label = f'{phase}_{target}_{family}_seed_{seed}'
            log = (REPORTS / f'{label}.log').open('a')
            command = [sys.executable, '-u', str(STORE / 'frozen_runner.py'), '--worker', phase,
                       '--target', target, '--family', family, '--seed', str(seed)]
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            active.append((p, log, label))
            print('START', label, p.pid, flush=True)
        for item in active[:]:
            p, log, label = item
            status = p.poll()
            if status is not None:
                active.remove(item)
                log.close()
                if status:
                    for other, _, _ in active:
                        other.terminate()
                    raise RuntimeError(f'Worker failed: {label}, exit={status}')
        time.sleep(2)


def budget_guards():
    triggered = []
    for target, family, mode in itertools.product(TARGETS, FAMILIES, ['local', 'fine']):
        lr = read(CONFIGS / f'lr_{target}_{family}_{mode}.json')['lr']
        folder = RUNS / f'{target}_{family}_{mode}_lr_{lr:g}_steps_800_seed_0'
        result = read(folder / 'training_complete.json')
        curve = {v['step']: v['validation']['macro']['total'] for v in result['history']}
        if result['best_step'] == 800 and curve[600] - curve[800] > .002:
            triggered.append(dict(target=target, family=family, mode=mode, lr=lr))
    c.save(CONFIGS / 'budget_guard.json', dict(triggered=triggered, selection='Validation only; separate1600-step sensitivity, not a replacement.'))
    for job in triggered:
        for seed in SEEDS:
            fit(job['target'], job['family'], seed, job['mode'], job['lr'], 1600)


def master(workers):
    initialize()
    if not (REPORTS / 'preflight.json').exists():
        preflight()
    log = (REPORTS / 'etas_training.log').open('a')
    etas = subprocess.Popen(['/root/autodl-tmp/venvs/em_eqf-etas/bin/python', '-u', str(STORE / 'frozen_runner.py'), '--etas'], stdout=log, stderr=subprocess.STDOUT)
    jobs_run([('source', t, f, s) for t,f,s in itertools.product(TARGETS, FAMILIES, SEEDS)], workers)
    jobs_run([('target', t, f, 0) for t,f in itertools.product(TARGETS, FAMILIES)], workers)
    jobs_run([('target', t, f, s) for t,f,s in itertools.product(TARGETS, FAMILIES, [1, 2])], workers)
    budget_guards()
    status = etas.wait()
    log.close()
    if status:
        raise RuntimeError('ETAS fitting failed; inspect etas_training.log')
    c.save(REPORTS / 'training_complete.json', dict(complete=True, test_scored=False))
    print('TARGET_TPP_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--etas', action='store_true')
    parser.add_argument('--worker', choices=['source', 'target'])
    parser.add_argument('--target', choices=TARGETS)
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.initialize:
        initialize()
    elif args.preflight:
        preflight()
    elif args.etas:
        etas_fit()
    elif args.worker == 'source':
        fit(args.target, args.family, args.seed, 'source', 3e-4, 6400)
    elif args.worker == 'target':
        target_worker(args.target, args.family, args.seed)
    elif args.run:
        master(args.workers)
    else:
        parser.error('Choose an action')
