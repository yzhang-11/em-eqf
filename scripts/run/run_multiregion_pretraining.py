"""Controlled leave-one-catalog-out pilot for the opt-in regional MixerTPP."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import Batch
from src.models.builders import ModelBuilder
from src.models.tpp.region_mixer_tpp import RegionalMixerTPP

OUT = ROOT/'experiments/revision/multiregion_pretraining'
DEVICE = torch.device('cuda:0')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def settings():
    cfg = OmegaConf.load(ROOT/'experiments/revision/main/configs/pretrain_azdx_main_seed_0.yaml')
    cfg.use_b_updater = False
    cfg.use_ssm_filter = False
    cfg.use_double_precision = False
    cfg.loss_weights = {'time_weight':1., 'mag_weight':1., 'b_weight':0., 'b_smooth_weight':0.}
    cfg.loss_reduction = 'none'
    cfg.dropout = .1
    cfg.mixer_model_config.dropout_prob = .1
    cfg.mixer_model_config.input_proj_dropout = .1
    cfg.features_input_keys = ['mag', 'log_inter_times']
    cfg.mixer_model_config.input_dim = 2
    cfg.dataset = 'MultiRegion'
    cfg.catalog_cfg = {'root_dir':str(ROOT/'data/MultiRegion/catalogs'), 'mag_completeness':'per_sequence'}
    cfg.epochs = 0  # This runner uses explicit optimizer steps, recorded below.
    cfg.Mc = 2.  # Legacy constructor placeholder; the regional head uses batch Mc.
    # Placeholder legacy defaults; every regional sequence overrides these.
    for key, value in dict(tau_mean=1., tau_min=0., tau_max=1., tau_q05=.1,
                           tau_q025=.05, mag_mean=2., time_max=1.,
                           richter_b_mle=1., mag_completeness=2.).items():
        cfg[key] = value
    return cfg


def build(condition, seed):
    torch.manual_seed(seed)
    cfg = settings()
    old = ModelBuilder.by_name('mixer_tpp')()(cfg, DEVICE)
    model = RegionalMixerTPP(old, cfg, condition_regions=condition).to(DEVICE)
    with torch.no_grad():
        # Same unit-mean exponential initialization for every comparison.
        model.hypernet_time.weight.normal_(std=.001)
        model.hypernet_time.bias[:2*model.num_components].fill_(math.log(math.expm1(1.)))
        model.hypernet_time.bias[2*model.num_components:].zero_()
        model.hypernet_mag[0].weight.zero_()
        model.hypernet_mag[0].bias.fill_(1.)
    # Extra conditioner initialization must not change subsequent dropout RNG.
    torch.manual_seed(seed + 1000)
    return model


@torch.no_grad()
def evaluate(model, sequences):
    model.eval()
    values = {'time':[], 'mag':[], 'total':[]}
    for left in range(0, len(sequences), 16):
        batch = Batch.from_list(sequences[left:left+16]).to(DEVICE)
        losses = model.nll_loss(batch, reduction='per_event')
        for key in values:
            values[key].extend(losses[key].cpu().tolist())
    if not values['total']:
        raise ValueError('No evaluation windows')
    if not all(np.isfinite(x).all() for x in values.values()):
        raise ValueError('Nonfinite evaluation loss')
    return {key:float(np.mean(x)) for key,x in values.items()}, values


def fit(data, heldout, sources, condition, variant, steps, seed, batch_size,
        init=None, fewshot=False):
    name = f'{heldout}__{variant}__seed_{seed}'
    folder = OUT/'runs'/name
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(heldout=heldout, sources=sources, condition_regions=condition,
                  steps=steps, seed=seed, batch_size=batch_size, fewshot=fewshot,
                  fewshot_scored_events=512 if fewshot else 0,
                  metadata_calibration_events=256, learning_rate=3e-4 if not fewshot else 1e-4,
                  dataset_sha256=DATA_SHA, initialization='source_pretrained' if init else 'scratch',
                  initialization_sha256=hashlib.sha256(Path(init).read_bytes()).hexdigest() if init else None,
                  backbone=OmegaConf.to_container(settings(), resolve=True),
                  validation='none; final update' if fewshot else 'source regions only')
    config_path = OUT/'configs'/f'{name}.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), config_path)
    metrics_path = folder/'metrics.json'
    if metrics_path.exists():
        result = json.loads(metrics_path.read_text())
        if result['config'] != config:
            raise ValueError(f'Existing run has different settings: {name}')
        return result, folder/'best_model.pth'
    model = build(condition, seed)
    if init:
        model.load_state_dict(torch.load(init, map_location=DEVICE, weights_only=False)['model_state_dict'])
    pools = {heldout:data[heldout]['train'][:4]} if fewshot else {s:data[s]['train'] for s in sources}
    if any(not pool for pool in pools.values()):
        raise ValueError('Empty training pool')
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.01)
    best = math.inf
    history = []
    started = time.time()
    for step in range(1, steps+1):
        model.train()
        selected = [rng.choice(pools[rng.choice(list(pools))]) for _ in range(batch_size)]
        batch = Batch.from_list(selected).to(DEVICE)
        losses = model.nll_loss(batch, reduction='per_event')
        loss = losses['total'].mean()
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite training loss in {name} step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad_norm):
            raise ValueError(f'Nonfinite gradient in {name} step {step}')
        warmup = min(1., step/max(1, steps//10))
        decay = .5*(1+math.cos(math.pi*step/steps))
        for group in optimizer.param_groups:
            group['lr'] = config['learning_rate']*warmup*max(.05, decay)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            validation = (float(loss) if fewshot else
                          float(np.mean([evaluate(model, data[s]['val'])[0]['total'] for s in sources])))
            row = dict(step=step, train_nll=float(loss), validation_nll=validation,
                       elapsed_seconds=time.time()-started)
            history.append(row)
            print(json.dumps(dict(run=name, **row)), flush=True)
            if fewshot or validation < best:
                best = validation
                torch.save(dict(model_state_dict=model.state_dict(), config=config, step=step), folder/'best_model.pth')
            save_json(folder/'progress.json', dict(config=config, history=history))
    model.load_state_dict(torch.load(folder/'best_model.pth', map_location=DEVICE, weights_only=False)['model_state_dict'])
    metrics, windows = evaluate(model, data[heldout]['test'])
    result = dict(name=name, config=config, metrics=metrics, test_windows=len(data[heldout]['test']),
                  elapsed_seconds=time.time()-started,
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  checkpoint_sha256=hashlib.sha256((folder/'best_model.pth').read_bytes()).hexdigest())
    save_json(folder/'test_window_nll.json', windows)
    save_json(metrics_path, result)
    print(json.dumps(dict(completed=name, **metrics)), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return result, folder/'best_model.pth'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--fewshot-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--heldout', choices=['all','japan_kyushu','italy','new_zealand'], default='all')
    parser.add_argument('--mode', choices=['full','core'], default='full',
                        help='core repeats only pooled and conditioned models for seed checks')
    args = parser.parse_args()
    path = ROOT/'data/MultiRegion/catalogs/pilot_2007_2019.pt'
    global DATA_SHA
    DATA_SHA = hashlib.sha256(path.read_bytes()).hexdigest()
    data = torch.load(path, map_location='cpu', weights_only=False)
    torch.set_num_threads(1)
    source_files = ['src/models/tpp/region_mixer_tpp.py', 'scripts/data/prepare_multiregion_pretraining.py',
                    'scripts/run/run_multiregion_pretraining.py', 'tests/model/test_region_mixer_tpp.py']
    manifest = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files}
    manifest_path = OUT/f'reports/source_manifest_seed_{args.seed}_20261104.json'
    if not manifest_path.exists():
        save_json(manifest_path, manifest)
    results = []
    heldouts = list(data) if args.heldout == 'all' else [args.heldout]
    for heldout in heldouts:
        sources = [s for s in data if s != heldout]
        if args.mode == 'full':
            for source in sources:
                result, _ = fit(data, heldout, [source], False, f'single_{source}', args.steps, args.seed, args.batch_size)
                results.append(result)
        for condition, variant in [(False,'pooled'), (True,'conditioned')]:
            result, checkpoint = fit(data, heldout, sources, condition, variant, args.steps, args.seed, args.batch_size)
            results.append(result)
        if args.mode == 'full':
            result, _ = fit(data, heldout, sources, True, 'conditioned_fewshot', args.fewshot_steps,
                            args.seed, args.batch_size, init=checkpoint, fewshot=True)
            results.append(result)
            result, _ = fit(data, heldout, [], True, 'scratch_fewshot', args.fewshot_steps,
                            args.seed, args.batch_size, fewshot=True)
            results.append(result)
        save_json(OUT/f'reports/pilot_results_seed_{args.seed}_20261104.json', results)
        rows = [dict(run=r['name'], heldout=r['config']['heldout'], sources=','.join(r['config']['sources']),
                     conditioned=r['config']['condition_regions'], fewshot=r['config']['fewshot'],
                     **r['metrics'], seconds=r['elapsed_seconds']) for r in results]
        pd.DataFrame(rows).to_csv(OUT/f'reports/pilot_results_seed_{args.seed}_20261104.csv', index=False)


if __name__ == '__main__':
    main()
