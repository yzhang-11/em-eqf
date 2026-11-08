"""Equal scored-event-budget data-size x sequence-length experiment.

All arms keep the existing stats-FiLM Mamba2 model and use identical core
validation/test events. Test data are evaluated only after all fits finish.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/run')]
from run_multidomain_pretraining import build_model, evaluate, load_multidomain_model, make_batch, model_settings, save_json
from src.data import Sequence
from src.models.tpp.region_mixer_tpp import attach_region, RegionalGridGR

STUDY = ROOT/'experiments/revision/multiregion_pretraining'
REPORTS = STUDY/'reports/long_sequence_20261105'
CONFIGS = STUDY/'configs/long_sequence_20261105'
RUNS = STUDY/'runs/long_sequence_20261105'
STORE = Path('/root/autodl-fs/em_eqf_training/long_sequence_20261105/runs')
DATA = ROOT/'data/MultiRegion/catalogs/long_sequence_20261105.pt'
ARMS = ['core_short', 'expanded_short', 'core_long', 'expanded_long']
SEEDS = [0, 1, 2]
STEPS = 1500
EVERY = 100


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sequence(source, left, context, targets, metadata, region_id):
    right = left+context+targets
    arrival = source['times'][left:right]-source['times'][left-1]
    assert len(arrival) == context+targets and np.all(np.diff(arrival) > 0)
    inter = np.diff(np.r_[0., arrival, arrival[-1]])
    seq = Sequence(torch.from_numpy(inter), t_nll_start=float(arrival[context-1]),
                   mag=torch.from_numpy(source['magnitude'][left:right].copy()))
    seq = attach_region(seq, metadata)
    seq.region_id = torch.full((len(seq),), region_id, dtype=torch.long)
    return seq


def pools(data, length):
    context, targets = (64, 128) if length == 'short' else (256, 1024)
    result = {}
    for region, spec in data['regions'].items():
        n = spec['train_targets']
        # The long window equals eight adjacent short scored windows.
        starts = range(257, 257+n, targets)
        result[region] = [sequence(spec['splits']['train'], start-context, context, targets,
                                    spec['metadata'], data['vocabulary'][region]) for start in starts]
        assert len(result[region])*targets == n
    return result


def evaluation_pools(data, split):
    result = {}
    for region, spec in data['regions'].items():
        if not spec['core']:
            continue
        source = spec['splits'][split]
        result[region] = [sequence(source, start-256, 256, 128, spec['metadata'], data['vocabulary'][region])
                          for start in range(257, len(source['times'])-128+1, 128)]
        assert result[region], (region, split)
    return result


def plan(data):
    CONFIGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    if not RUNS.exists():
        RUNS.symlink_to(STORE, target_is_directory=True)
    config = dict(protocol='data_sequence_factorial_20261105_v1', arms=ARMS, seeds=SEEDS,
        max_updates=STEPS, validation_every=EVERY, learning_rate=3e-4, optimizer='AdamW',
        weight_decay=.01, gradient_clip=1., scheduler='warmup10percent_cosine_floor5percent',
        model='stats_film Mamba2Rotary; unchanged architecture', training_initialization='scratch; paired seed',
        short=dict(context=64, targets=128, total_events=192, batch_size=16),
        long=dict(context=256, targets=1024, total_events=1280, batch_size=2),
        scored_events_per_update=2048, max_scored_event_exposures=STEPS*2048,
        sampling='Uniform region, then uniform window, with replacement, per sample. Batch sizes differ.',
        evaluation=dict(context=256, targets=128, stride=128, shared_target_events=True),
        checkpoint_selection='single checkpoint, core three-region validation macro total NLL',
        split='train before 2014; validation 2014-2015; test 2016-2019',
        test_selection=False, new_regions='Training only; all records before 2014',
        dataset_sha256=sha(DATA),
        core_regions=sorted(r for r, spec in data['regions'].items() if spec['core']),
        expanded_regions=sorted(data['regions']))
    path = CONFIGS/'study_plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        save_json(path, config)
    return config


def fit(data, train, validation, study, arm, seed):
    length = 'long' if arm.endswith('long') else 'short'
    settings = study[length]
    sources = study['core_regions'] if arm.startswith('core_') else study['expanded_regions']
    name = f'{arm}_seed_{seed}'
    folder = RUNS/name
    cfg = dict(protocol=study['protocol'], name=name, method='stats_film', seed=seed,
        sources=sources, targets=study['core_regions'], region_id_map=data['vocabulary'],
        backbone=OmegaConf.to_container(model_settings('stats_film'), resolve=True),
        dataset_sha256=study['dataset_sha256'], optimizer_steps=STEPS, validation_every=EVERY,
        learning_rate=study['learning_rate'], batch_size=settings['batch_size'],
        context_events=settings['context'], scored_events_per_window=settings['targets'],
        scored_events_per_update=2048, weight_decay=.01, initialization=None,
        train_unique_scored_events=sum(data['regions'][r]['train_targets'] for r in sources))
    complete_path = folder/'training_complete.json'
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        assert complete['config'] == cfg and sha(folder/'best_model.pth') == complete['checkpoint_sha256']
        print('REUSE', name, flush=True)
        return complete
    folder.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), CONFIGS/f'{name}.yaml')
    model = build_model('stats_film', seed, data['vocabulary'], cfg['backbone'])
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = random.Random(seed)
    best, history = math.inf, []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    counts = dict.fromkeys(sources, 0)
    for step in range(1, STEPS+1):
        model.train()
        selected = []
        for _ in range(cfg['batch_size']):
            region = rng.choice(sources)
            counts[region] += settings['targets']
            selected.append(rng.choice(train[region]))
        batch = make_batch(selected)
        assert int(batch.nll_event_mask.sum()) == 2048
        losses = model.nll_loss(batch, reduction='per_event')
        loss = losses['total'].mean()
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss {name}:{step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError(f'Nonfinite gradient {name}:{step}')
        lr = 3e-4*min(1., step/(STEPS//10))*max(.05, .5*(1+math.cos(math.pi*step/STEPS)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if step % EVERY == 0:
            metrics, _ = evaluate(model, validation)
            row = dict(step=step, train_nll=float(loss), validation=metrics,
                       elapsed_seconds=time.time()-started)
            history.append(row)
            if metrics['macro']['total'] < best:
                best = metrics['macro']['total']
                temporary = folder/'best_model.tmp'
                torch.save(dict(model_state_dict=model.state_dict(), config=cfg, step=step, validation=metrics), temporary)
                temporary.replace(folder/'best_model.pth')
            save_json(folder/'progress.json', dict(config=cfg, history=history,
                                                   initial_state_sha256=initial_hash))
            print(json.dumps(dict(run=name, step=step, validation=metrics['macro'],
                                  seconds=round(row['elapsed_seconds'], 1))), flush=True)
    payload = torch.load(folder/'best_model.pth', map_location='cpu', weights_only=False)
    complete = dict(config=cfg, checkpoint=str(folder/'best_model.pth'), checkpoint_sha256=sha(folder/'best_model.pth'),
        best_step=payload['step'], validation=payload['validation'], initial_state_sha256=initial_hash,
        elapsed_seconds=time.time()-started, scored_event_exposures=counts,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), history=history)
    save_json(complete_path, complete)
    print(json.dumps(dict(completed=name, best_step=payload['step'])), flush=True)
    del model, optimizer, payload
    torch.cuda.empty_cache()
    return complete


def finalize(data, study):
    expected = [f'{arm}_seed_{seed}' for seed in SEEDS for arm in ARMS]
    completed = {name: json.loads((RUNS/name/'training_complete.json').read_text()) for name in expected}
    for seed in SEEDS:
        assert len({completed[f'{arm}_seed_{seed}']['initial_state_sha256'] for arm in ARMS}) == 1
    test = evaluation_pools(data, 'test')
    rows = []
    for name, complete in completed.items():
        folder = RUNS/name
        assert sha(folder/'best_model.pth') == complete['checkpoint_sha256']
        model, _ = load_multidomain_model(folder/'best_model.pth')
        metrics, windows = evaluate(model, test)
        save_json(folder/'metrics.json', dict(**complete, test=metrics))
        save_json(folder/'test_window_nll.json', windows)
        for region in [*study['core_regions'], 'macro']:
            result = metrics['macro'] if region == 'macro' else metrics['regions'][region]
            rows.append(dict(arm=name.rsplit('_seed_', 1)[0], seed=complete['config']['seed'], region=region,
                             best_step=complete['best_step'], **result, checkpoint=str(folder/'best_model.pth')))
        del model
    aggregates = []
    for arm in ARMS:
        for region in [*study['core_regions'], 'macro']:
            group = [row for row in rows if row['arm'] == arm and row['region'] == region]
            aggregates.append(dict(arm=arm, region=region, **{k:dict(mean=statistics.mean(r[k] for r in group),
                sd=statistics.stdev(r[k] for r in group), n=3) for k in ['time', 'mag', 'total']}))
    with (REPORTS/'all_results.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(plan=study, completed_fits=len(completed), results=aggregates,
                   paired_initializations_verified=True,
                   test_windows={r:len(v) for r,v in test.items()})
    save_json(REPORTS/'summary.json', summary)
    info = json.loads((REPORTS/'data.json').read_text())
    lines = ['# Larger-data and longer-sequence experiment (2026-11-05)', '',
        'NLL is nats/event, lower is better. Mean +/- sample SD over three seeds. All 12 fits finished before test evaluation.', '',
        f"Training scored event universe: core {info['core_train_scored_events']}; expanded {info['expanded_train_scored_events']}.",
        'Short: 64 context + 128 targets, batch 16. Long: 256 context + 1024 targets, batch 2.',
        'All arms: 1500 updates, 2048 scored events/update, validation every 100 updates, one core-validation-selected checkpoint.',
        'Model: unchanged stats-FiLM Mamba2Rotary, trained from scratch. Paired initial weights verified.',
        'Evaluation: identical core-region target events, 256 context + 128 targets. This differs from the old pilot evaluation context (64).', '',
        '| Region | core_short | expanded_short | core_long | expanded_long |',
        '|---|---:|---:|---:|---:|']
    for region in [*study['core_regions'], 'macro']:
        cells = []
        for arm in ARMS:
            m = next(r for r in aggregates if r['arm']==arm and r['region']==region)['total']
            cells.append(f"{m['mean']:.6f} +/- {m['sd']:.6f}")
        lines.append('| '+region+' | '+' | '.join(cells)+' |')
    lines += ['', '| Arm | Macro time NLL | Macro magnitude NLL |', '|---|---:|---:|']
    for arm in ARMS:
        row = next(r for r in aggregates if r['arm']==arm and r['region']=='macro')
        lines.append(f"| {arm} | {row['time']['mean']:.6f} | {row['mag']['mean']:.6f} |")
    lines += ['', 'Limits: catalog thresholds are fixed research choices, not completeness certificates. Added data differ in both region and historical span.',
              'Equal scored-event budgets do not imply equal FLOPs: long windows process more context and use a smaller batch.',
              'Uniform region sampling reduces the fraction of updates from each original region when more regions are added.',
              'This is a known-domain likelihood experiment; it does not establish unseen-domain transfer or long-horizon simulation quality.', '',
              'Artifacts:', str(CONFIGS), str(RUNS), str(REPORTS), '',
              'Reproduce: source /root/autodl-tmp/venvs/em_eqf-cu128/env.sh; python scripts/run/run_long_multiregion.py --all', '']
    (REPORTS/'results.md').write_text('\n'.join(lines))
    print('FINAL', json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    assert args.all or args.seed is not None or args.finalize
    torch.set_num_threads(1)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        data = torch.load(DATA, map_location='cpu', weights_only=False)
        study = plan(data)
        if args.all or args.seed is not None:
            train = {length:pools(data,length) for length in ['short','long']}
            validation = evaluation_pools(data, 'val')
            for seed in SEEDS if args.all else [args.seed]:
                for arm in ARMS:
                    fit(data, train['long' if arm.endswith('long') else 'short'], validation, study, arm, seed)
        if args.all or args.finalize:
            finalize(data, study)


if __name__ == '__main__':
    main()
