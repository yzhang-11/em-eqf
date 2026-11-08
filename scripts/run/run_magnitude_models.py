"""Validation-driven search of likelihood-only magnitude heads on frozen causal contexts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import lfilter
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
from src.models.tpp.magnitude_density import MagnitudeDensityHead

OUT = ROOT / 'experiments/revision/magnitude_models'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def load_data(region, variant, seed, mode):
    with np.load(OUT/'features'/f'{region}_{variant}_events.npz') as d:
        data = {k: d[k].copy() for k in d.files}
    with np.load(OUT/'features'/f'{region}_{variant}_seed_{seed}.npz') as d:
        context = d['context'].copy()
        data['old_logp'] = d['old_logp'].copy()
    if mode == 'static':
        features = np.zeros((len(context), 1), dtype=np.float32)
    elif mode == 'history':
        features = data['history']
    elif mode == 'context_history':
        features = np.concatenate([context, data['history']], axis=-1)
    elif mode.startswith('context_'):
        length = int(mode.split('_')[1])
        alpha = 1./length
        features = lfilter([alpha], [1., -(1.-alpha)], context, axis=0)
        features /= (1. - (1.-alpha) ** np.arange(1, len(context)+1))[:, None]
        features = features.astype(np.float32)
    else:
        assert mode == 'context'
        features = context
    data['features'] = features
    return data


def train_fit(region, variant, seed, recipe, stage):
    family, components, mode = recipe['family'], recipe['components'], recipe['features']
    name = f"{stage}_{region}_{variant}_{family}{components}_{mode}_seed_{seed}"
    run = OUT/'runs'/name
    run.mkdir(parents=True, exist_ok=True)
    path = run/'job_status.json'
    config = dict(region=region, variant=variant, seed=seed, family=family, components=components,
                  features=mode, hidden=32, maximum_epochs=400, patience=35,
                  learning_rate=.003, weight_decay=0., dropout=0.,
                  loss='negative log reported-bin probability only', b_weight=0., b_smooth_weight=0.,
                  encoder='frozen; cached causal contexts', stage=stage)
    config_path = OUT/'configs'/f'{name}.json'
    if path.exists():
        status = json.loads(path.read_text())
        if status['state'] == 'completed':
            assert json.loads(config_path.read_text()) == config
            return status
        raise RuntimeError('Inspect incomplete run: ' + name)
    save(config_path, config)
    save(path, dict(state='running', name=name, started_utc=datetime.now(timezone.utc).isoformat()))
    begin = time.monotonic()
    torch.manual_seed(20261104+seed)
    np.random.seed(20261104+seed)
    data = load_data(region, variant, seed, mode)
    train, val = data['split'] == 1, data['split'] == 2
    assert train.any() and val.any()
    mean = data['features'][train].mean(0)
    std = np.maximum(data['features'][train].std(0), 1e-5)
    features = (data['features']-mean)/std
    device = torch.device('cuda:0')
    xtr = torch.tensor(features[train], device=device)
    xva = torch.tensor(features[val], device=device)
    ytr = torch.tensor(data['magnitude'][train], device=device)
    yva = torch.tensor(data['magnitude'][val], device=device)
    model = MagnitudeDensityHead(0 if mode == 'static' else features.shape[1], family, components,
                                 float(data['Mc']), float(data['step']), hidden=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
    best, best_epoch, best_state, stale = float('inf'), 0, None, 0
    curve = []
    for epoch in range(config['maximum_epochs']+1):
        if epoch:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = -model(xtr).log_prob(ytr).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'{name}: nonfinite loss at {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        if epoch % 2:
            continue
        model.eval()
        with torch.inference_mode():
            train_nll = float(-model(xtr).log_prob(ytr).mean())
            validation = float(-model(xva).log_prob(yva).mean())
        curve.append(dict(epoch=epoch, train=train_nll, validation=validation))
        if validation < best - 1e-6:
            best, best_epoch, stale = validation, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 2
        if stale >= config['patience']:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint = run/'best_head.pth'
    torch.save(dict(state_dict=best_state, normalization_mean=mean, normalization_std=std,
                    config=config, Mc=float(data['Mc']), step=float(data['step']), best_epoch=best_epoch,
                    validation_nll=best, input_dim=model.input_dim,
                    base_checkpoint=str(ROOT/'experiments/revision/main/runs'/f'generation_{region}_{variant}_seed_{seed}'/'best_model_1.pth')),
               checkpoint)
    pd.DataFrame(curve).to_csv(run/'learning_curve.csv', index=False)
    original_val = float(-data['old_logp'][val].mean())
    status = dict(state='completed', name=name, region=region, variant=variant, seed=seed,
                  **recipe, best_epoch=best_epoch, validation_nll=best, original_validation_nll=original_val,
                  validation_gain=original_val-best, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  elapsed_seconds=round(time.monotonic()-begin, 3), test_used_for_selection=False)
    save(path, status)
    print(json.dumps(status), flush=True)
    return status


def screen():
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        goal='Improved proper magnitude probability without b-reference or b-smoothness auxiliary penalties',
        selection='validation NLL only; candidate test scores withheld until validation selection is frozen',
        scoring='same reported-bin masses for original and candidates; SWC .1, SCSN .01, common upper bound 10',
        features='past-only encoder context or causal running summaries; no current target mark/interval',
        success='paired test gain over original; five fits and common calendar block resampling; report Mc sensitivity and calibration',
        optimizer='Adam; no weight decay, no dropout, no auxiliary loss; finite-gradient clipping only',
        source_files=[dict(path=str(ROOT/p),sha256=digest(ROOT/p)) for p in [
            'src/models/tpp/magnitude_density.py','scripts/data/prepare_magnitude_features.py',
            'scripts/run/run_magnitude_models.py']])
    path = OUT/'plan.json'
    if path.exists():
        old = json.loads(path.read_text())
        for row in old['source_files']:
            assert digest(row['path']) == row['sha256'], row['path']
    else:
        save(path, plan)
    recipes = []
    for mode in ['static', 'context', 'context_64', 'history', 'context_history']:
        recipes += [dict(family='gr', components=1, features=mode),
                    dict(family='weibull', components=1, features=mode),
                    dict(family='kumaraswamy', components=2, features=mode)]
    results = []
    for region in ['swc', 'scsn']:
        for recipe in recipes:
            save(OUT/'reports/search_status.json', dict(state='screening', region=region, recipe=recipe,
                                                      completed=len(results), total=2*len(recipes)))
            results.append(train_fit(region, 'main', 0, recipe, 'screen'))
            pd.DataFrame(results).to_csv(OUT/'reports/screen_validation_scores.csv', index=False)
    selected = {}
    for region in ['swc', 'scsn']:
        subset = sorted([r for r in results if r['region'] == region], key=lambda r: r['validation_nll'])
        flexible = [r for r in subset if r['family'] != 'gr'][:2]
        gr = next(r for r in subset if r['family'] == 'gr')
        selected[region] = [{k: r[k] for k in ['family', 'components', 'features']} for r in flexible+[gr]]
    save(OUT/'configs/confirmation_recipes.json', selected)
    save(OUT/'reports/search_status.json', dict(state='screen_completed', fits=len(results), confirmation=selected))


def confirm():
    recipes = json.loads((OUT/'configs/confirmation_recipes.json').read_text())
    results = []
    for region in ['swc', 'scsn']:
        for variant in ['main', 'sensitivity']:
            for recipe in recipes[region]:
                for seed in range(5):
                    save(OUT/'reports/search_status.json', dict(state='confirming', region=region, variant=variant,
                        recipe=recipe, seed=seed, completed=len(results), total=60))
                    results.append(train_fit(region, variant, seed, recipe, 'confirm'))
                    pd.DataFrame(results).to_csv(OUT/'reports/confirmation_validation_scores.csv', index=False)
    chosen = {}
    for region in ['swc', 'scsn']:
        aggregates = []
        for recipe in recipes[region]:
            rows = [r for r in results if r['region'] == region and r['variant'] == 'main'
                    and all(r[k] == v for k,v in recipe.items())]
            assert len(rows) == 5
            aggregates.append(dict(**recipe, validation_nll=float(np.mean([r['validation_nll'] for r in rows])),
                                   mean_gain=float(np.mean([r['validation_gain'] for r in rows]))))
        flexible = sorted([r for r in aggregates if r['family'] != 'gr'], key=lambda r:r['validation_nll'])
        chosen[region] = dict(selected=flexible[0], controls=aggregates)
    selection = dict(state='frozen_before_candidate_test_scoring', created_utc=datetime.now(timezone.utc).isoformat(),
                     recipes=chosen, rule='lowest five-fit main-validation mean; sensitivity not used to select architecture',
                     training_source_sha256=digest(Path(__file__)))
    save(OUT/'selection.json', selection)
    save(OUT/'reports/search_status.json', dict(state='validation_selection_frozen', fits=len(results)))
    print(json.dumps(selection), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirm'])
    options = parser.parse_args()
    torch.set_num_threads(1)
    {'screen':screen, 'confirm':confirm}[options.stage]()
