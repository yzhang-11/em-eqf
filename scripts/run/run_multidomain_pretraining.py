"""Global autoregressive models with categorical region embeddings.

Known-domain forecasting and unseen-domain adaptation are separate protocols.
Architecture selection uses validation likelihood only. Old pilot files stay put.
"""
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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import Batch
from src.models.builders import ModelBuilder
from src.models.tpp.region_mixer_tpp import RegionalMixerTPP
from src.models.tpp.global_mixer_tpp import GlobalMixerTPP
from run_multiregion_pretraining import settings, save_json

STUDY = ROOT/'experiments/revision/multiregion_pretraining'
RUNS = STUDY/'runs/multidomain_id'
REPORTS = STUDY/'reports/multidomain_id'
CONFIGS = STUDY/'configs/multidomain_id'
DEVICE = torch.device('cuda:0')
METHODS = {
    'stats_film': dict(condition_stats=True, conditioning_mode='film', use_id=False),
    'stats_concat': dict(condition_stats=True, conditioning_mode='concat', use_id=False),
    'id_concat': dict(condition_stats=False, conditioning_mode='concat', use_id=True),
    'stats_id_concat': dict(condition_stats=True, conditioning_mode='concat', use_id=True),
    'stats_id_film': dict(condition_stats=True, conditioning_mode='film', use_id=True),
    'id_input': dict(use_id=True, input_embedding=True),
}
ID_METHODS = ['id_concat', 'stats_id_concat', 'stats_id_film', 'id_input']
TRANSFER_METHODS = ['stats_film', 'stats_id_concat', 'stats_id_film', 'id_input']


def model_settings(method):
    cfg = settings()
    if METHODS[method].get('input_embedding'):
        cfg.mixer_model_config.input_dim = 10  # two observed features + eight static-ID coordinates
    return cfg


def build_model(method, seed, vocabulary, backbone=None, device=DEVICE):
    torch.manual_seed(seed)
    cfg = model_settings(method) if backbone is None else OmegaConf.create(backbone)
    base = ModelBuilder.by_name('mixer_tpp')()(cfg, device)
    options = dict(METHODS[method])
    use_id = options.pop('use_id')
    use_input = options.pop('input_embedding', False)
    if use_input:
        model = GlobalMixerTPP(base, cfg, num_regions=len(vocabulary)-1, region_embedding_dim=8).to(device)
    else:
        model = RegionalMixerTPP(base, cfg, num_regions=len(vocabulary)-1 if use_id else 0,
                                 region_embedding_dim=8, **options).to(device)
    with torch.no_grad():
        model.hypernet_time.weight.normal_(std=.001)
        model.hypernet_time.bias[:2*model.num_components].fill_(math.log(math.expm1(1.)))
        model.hypernet_time.bias[2*model.num_components:].zero_()
        model.hypernet_mag[0].weight.zero_()
        model.hypernet_mag[0].bias.fill_(1.)
    torch.manual_seed(seed+1000)
    return model


def load_multidomain_model(path, device=DEVICE):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload['config']
    model = build_model(config['method'], config['seed'], config['region_id_map'],
                        backbone=config['backbone'], device=device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model.eval(), config


def load_default_multidomain_model(device=DEVICE):
    spec = json.loads((REPORTS/'default_model.json').read_text())
    path = Path(spec['checkpoint'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['checkpoint_sha256']:
        raise ValueError('Default checkpoint checksum mismatch')
    return load_multidomain_model(path, device=device)


def make_batch(sequences, unknown=False):
    batch = Batch.from_list(sequences).to(DEVICE)
    if unknown:
        batch.region_id.zero_()
    return batch


@torch.no_grad()
def evaluate(model, pools, unknown=False):
    model.eval()
    metrics, windows = {}, {}
    for region, sequences in pools.items():
        values = {k:[] for k in ['time','mag','total']}
        if not sequences:
            raise ValueError(f'Empty evaluation pool: {region}')
        for left in range(0, len(sequences), 16):
            losses = model.nll_loss(make_batch(sequences[left:left+16], unknown), reduction='per_event')
            for k in values:
                values[k].extend(losses[k].cpu().tolist())
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError('Nonfinite evaluation likelihood')
        metrics[region] = {k:float(np.mean(v)) for k,v in values.items()}
        windows[region] = values
    macro = {k:float(np.mean([v[k] for v in metrics.values()])) for k in ['time','mag','total']}
    return dict(macro=macro, regions=metrics), windows


def fit(data, vocabulary, dataset_sha, name, method, sources, seed, steps,
        batch_size, targets, *, init=None, adaptation=None):
    folder = RUNS/name
    folder.mkdir(parents=True, exist_ok=True)
    cfg = dict(protocol='multidomain_id_v1', name=name, method=method, sources=sources,
               targets=targets, seed=seed, optimizer_steps=steps, batch_size=batch_size,
               region_id_map=vocabulary, embedding_dim=8,
               dataset_sha256=dataset_sha, metadata_calibration_events=256,
               adaptation=adaptation, learning_rate=1e-3 if adaptation=='embedding' else 1e-4 if adaptation else 3e-4,
               initialization=str(init) if init else None,
               initialization_sha256=hashlib.sha256(Path(init).read_bytes()).hexdigest() if init else None,
               adaptation_train_scored_events=512 if adaptation else 0,
               adaptation_validation_scored_events=256 if adaptation else 0,
               unknown_id_policy='zero shared fallback before target adaptation',
               validation='target training-period windows 5-6' if adaptation else 'source validation periods only',
               backbone=OmegaConf.to_container(model_settings(method), resolve=True))
    if (folder/'metrics.json').exists():
        saved = json.loads((folder/'metrics.json').read_text())
        if saved['config'] != cfg:
            raise ValueError(f'Existing run has different settings: {name}')
        return saved
    CONFIGS.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), CONFIGS/f'{name}.yaml')
    model = build_model(method, seed, vocabulary, cfg['backbone'])
    if init:
        payload = torch.load(init, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload['model_state_dict'], strict=True)
    hook = None
    if adaptation:
        target = targets[0]
        if model.region_embedding is not None and init:
            model.initialize_region_embedding(vocabulary[target], [vocabulary[s] for s in sources])
        for key, parameter in model.named_parameters():
            parameter.requires_grad_(adaptation == 'scratch' or
                                     (adaptation == 'heads' and not key.startswith('base_model.encoder.')) or
                                     (adaptation == 'embedding' and key == 'region_embedding.weight'))
        if model.region_embedding is not None:
            mask = torch.zeros_like(model.region_embedding.weight)
            mask[vocabulary[target]] = 1
            hook = model.region_embedding.weight.register_hook(lambda gradient: gradient*mask)
        pools = {target:data[target]['train'][:4]}
        validation_pools = {target:data[target]['train'][4:6]}
    else:
        pools = {s:data[s]['train'] for s in sources}
        validation_pools = {s:data[s]['val'] for s in sources}
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError('No trainable adaptation parameters')
    # Embedding-table rows not in the target must not move through weight decay.
    optimizer = torch.optim.AdamW(trainable, lr=cfg['learning_rate'], weight_decay=0. if adaptation else .01)
    rng = random.Random(seed)
    best, history = math.inf, []
    started = time.time()
    interval = max(1, steps//3)
    for step in range(1, steps+1):
        model.eval() if adaptation else model.train()
        selected = [rng.choice(pools[rng.choice(list(pools))]) for _ in range(batch_size)]
        loss = model.nll_loss(make_batch(selected), reduction='per_event')['total'].mean()
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss: {name}, step={step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        if not torch.isfinite(norm):
            raise ValueError(f'Nonfinite gradient: {name}, step={step}')
        lr = cfg['learning_rate']*min(1.,step/max(1,steps//10))*max(.05,.5*(1+math.cos(math.pi*step/steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if step % interval == 0 or step == steps:
            validation, _ = evaluate(model, validation_pools)
            row = dict(step=step, train_nll=float(loss), validation_nll=validation['macro']['total'],
                       elapsed_seconds=time.time()-started)
            history.append(row)
            if row['validation_nll'] < best:
                best = row['validation_nll']
                temporary = folder/'best_model.tmp'
                torch.save(dict(model_state_dict=model.state_dict(), config=cfg, step=step,
                                validation=validation), temporary)
                temporary.replace(folder/'best_model.pth')
            save_json(folder/'progress.json', dict(config=cfg, history=history))
            print(json.dumps(dict(run=name, **row)), flush=True)
    if hook is not None:
        hook.remove()
    payload = torch.load(folder/'best_model.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload['model_state_dict'])
    unknown = not adaptation and any(t not in sources for t in targets)
    tests, windows = evaluate(model, {t:data[t]['test'] for t in targets}, unknown=unknown)
    result = dict(config=cfg, validation=payload['validation'], test=tests,
                  best_step=payload['step'], test_uses_unknown_id=unknown,
                  elapsed_seconds=time.time()-started,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in trainable),
                  test_windows={t:len(data[t]['test']) for t in targets},
                  checkpoint=str(folder/'best_model.pth'),
                  checkpoint_sha256=hashlib.sha256((folder/'best_model.pth').read_bytes()).hexdigest())
    save_json(folder/'test_window_nll.json', windows)
    save_json(folder/'metrics.json', result)
    print(json.dumps(dict(completed=name, validation=best, test=tests['macro'])), flush=True)
    del model, optimizer, payload
    torch.cuda.empty_cache()
    return result


def select_known(results):
    available = {r['config']['method'] for r in results if r['config']['name'].startswith('known_')}
    means = {m:float(np.mean([r['validation']['macro']['total'] for r in results
                             if r['config']['name'].startswith('known_') and r['config']['method']==m]))
             for m in METHODS if m in available}
    method = min(means, key=means.get)
    reference = next(r for r in results if r['config']['name']==f'known_{method}_seed_0')
    spec = dict(method=method, selection='mean source/known-domain validation NLL over seeds; test not used',
                candidate_validation=means, seed=0, checkpoint=reference['checkpoint'],
                checkpoint_sha256=reference['checkpoint_sha256'],
                region_id_map=reference['config']['region_id_map'])
    save_json(REPORTS/'default_model.json', spec)
    OmegaConf.save(OmegaConf.create(spec), CONFIGS/'default.yaml')
    return spec


def completed_results(prefixes):
    rows=[]
    for path in RUNS.glob('*/metrics.json'):
        result=json.loads(path.read_text())
        if any(result['config']['name'].startswith(prefix) for prefix in prefixes):
            rows.append(result)
    return rows


def adaptation_name(target, method, mode):
    name=f'scratch_{target}_seed_0' if mode=='scratch' else f'adapt_{target}_{mode}_seed_0'
    path=RUNS/name/'metrics.json'
    if path.exists() and json.loads(path.read_text())['config']['method'] != method:
        return name.replace('_seed_0',f'_{method}_seed_0')
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['known','transfer','all'], default='all')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--updates-per-source', type=int, default=150)
    parser.add_argument('--adaptation-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--methods', nargs='+', choices=list(METHODS), default=list(METHODS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = ROOT/'data/MultiRegion/catalogs/pilot_2007_2019.pt'
    dataset_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    data = torch.load(path, map_location='cpu', weights_only=False)
    regions = sorted(data)
    vocabulary = {'<unknown>':0, **{name:i+1 for i,name in enumerate(regions)}}
    for name, splits in data.items():
        for sequences in splits.values():
            for seq in sequences:
                seq.region_id = torch.full((len(seq),), vocabulary[name], dtype=torch.long)
    save_json(REPORTS/'region_vocabulary.json', vocabulary)
    files=['src/models/tpp/region_mixer_tpp.py','src/models/tpp/global_mixer_tpp.py','scripts/run/run_multidomain_pretraining.py']
    save_json(REPORTS/'source_manifest_input_extension.json', {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})
    results=[]
    if args.stage in {'known','all'}:
        for seed in args.seeds:
            for method in args.methods:
                results.append(fit(data,vocabulary,dataset_sha,f'known_{method}_seed_{seed}',method,
                                   regions,seed,args.updates_per_source*len(regions),args.batch_size,regions))
            for region in regions:
                results.append(fit(data,vocabulary,dataset_sha,f'local_{region}_seed_{seed}','stats_film',
                                   [region],seed,args.updates_per_source,args.batch_size,[region]))
        known=completed_results(['known_','local_'])
        save_json(REPORTS/'known_results.json', known)
        if 0 in args.seeds:
            print(json.dumps(dict(selected_default=select_known(known))), flush=True)
    if args.stage in {'transfer','all'}:
        transferred=[]
        for target in regions:
            sources=[s for s in regions if s!=target]
            core=[]
            for seed in args.seeds:
                for method in [m for m in TRANSFER_METHODS if m in args.methods]:
                    result=fit(data,vocabulary,dataset_sha,f'heldout_{target}_{method}_seed_{seed}',method,
                               sources,seed,args.updates_per_source*len(sources),args.batch_size,[target])
                    transferred.append(result);core.append(result)
            core=completed_results([f'heldout_{target}_'])
            available={r['config']['method'] for r in core}
            means={m:float(np.mean([r['validation']['macro']['total'] for r in core
                                   if r['config']['method']==m])) for m in TRANSFER_METHODS if METHODS[m]['use_id'] and m in available}
            selected=min(means,key=means.get)
            save_json(REPORTS/f'transfer_selection_{target}.json',dict(method=selected,validation=means,
                      selection='source-only validation; target validation/test not used'))
            if 0 in args.seeds:
                parent=next(r for r in core if r['config']['method']==selected and r['config']['seed']==0)
                for mode in ['embedding','heads']:
                    transferred.append(fit(data,vocabulary,dataset_sha,adaptation_name(target,selected,mode),selected,
                                           sources,0,args.adaptation_steps,args.batch_size,[target],
                                           init=parent['checkpoint'],adaptation=mode))
                transferred.append(fit(data,vocabulary,dataset_sha,adaptation_name(target,selected,'scratch'),selected,
                                       [],0,args.adaptation_steps,args.batch_size,[target],adaptation='scratch'))
            save_json(REPORTS/'transfer_results.json', completed_results(['heldout_','adapt_','scratch_']))


if __name__ == '__main__':
    main()
