"""Paired calendar-block time likelihood comparisons of frozen fitted models."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
from src.data import Batch, Catalog
from src.catalogs.pathing import build_tpp_catalog_init_kwargs
from src.train.config_setup import load_and_prepare_model

BASE = ROOT/'experiments/revision/main'
FOLLOWUP = ROOT/'experiments/revision/generation_baselines'
OUT = FOLLOWUP/'reports/time_comparison'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    path = OUT/name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def time_nll(model, sequence, device):
    batch = Batch.from_list([sequence]).to(device)
    kwargs = dict(reduction='none')
    if 'return_dict' in inspect.signature(model.nll_loss).parameters:
        kwargs['return_dict'] = True
    with torch.no_grad():
        values = model.nll_loss(batch, **kwargs)
    assert isinstance(values, dict) and 'time' in values
    result = float(values['time'].sum().cpu())
    assert np.isfinite(result)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((FOLLOWUP/'frozen_manifest.json').read_text())
    plan = dict(calendar_segments=20, block_segments=[1, 2, 4], bootstrap_replicates=2000,
        component='time NLL only; includes compensator and censoring; totals with training penalties not compared',
        history='all observed events before each segment; same half-open target intervals',
        additivity_check='segment sums must reconstruct saved full-test time NLL within 2e-5 per day',
        resampling='shared calendar blocks; independently resampled five neural fits per family; fixed single ETAS fit',
        scope='conditional on catalog, fixed architecture, hyperparameters and fitted ETAS; no multiplicity adjustment',
        source_manifest_sha256=digest(FOLLOWUP/'frozen_manifest.json'), script_sha256=digest(__file__))
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text()) == plan
    else:
        save('plan.json', plan)
    save('status.json', dict(state='waiting_for_magnitude_test', gpu_preflight='pending'))
    while True:
        try:
            state = json.loads((BASE/'reports/magnitude_test/status.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
            continue
        if state['state'] == 'failed':
            raise RuntimeError('M-test replay failed; holding subsequent GPU comparison for inspection')
        if state['state'] == 'completed':
            break
        time.sleep(30)
    for group in ['source_files', 'configs', 'raw_sources', 'analysis_files']:
        for row in frozen[group]:
            assert digest(row['path']) == row['sha256'], row['path']
    assert digest(__file__) == plan['script_sha256']
    assert json.loads((FOLLOWUP/'queue_status.json').read_text())['state'] == 'completed'
    fits, intervals, checkpoint_records = [], [], []
    device = torch.device('cuda:0')
    for region in ['swc', 'scsn']:
        cfg = OmegaConf.load(BASE/'configs'/f'generation_{region}_main_seed_0.yaml')
        cls = Catalog.by_name(cfg.dataset+'-Standard')
        catalog = cls(**build_tpp_catalog_init_kwargs(cls, ROOT/'data'/cfg.dataset, cfg.catalog_cfg))
        sequence = catalog.test[0].double()
        start, end = float(sequence.t_nll_start), float(sequence.t_end)
        duration = end-start
        edges = np.linspace(start, end, 21)
        times = sequence.arrival_times.numpy()
        assert not np.isin(edges, times).any(), 'Boundary event requires explicit half-open extraction'
        segmented = []
        for a, b in zip(edges[:-1], edges[1:]):
            part = sequence.get_subsequence(sequence.t_start, b, reset_t_nll_to_end=True)
            part.t_end = float(b)
            part.t_nll_start = float(a)
            segmented.append(part)
        arrays = {}
        for family, parent, prefix, seeds in [
            ('EM-EQF', BASE, 'generation', range(5)),
            ('RECAST', FOLLOWUP, 'recast', range(5)),
            ('ETAS', FOLLOWUP, 'etas', range(1)),
        ]:
            scores = []
            for seed in seeds:
                name = f'{prefix}_{region}_main_seed_{seed}'
                folder = parent/'runs'/name
                checkpoint = folder/'best_model_1.pth'
                sha = digest(checkpoint)
                status = json.loads((folder/'job_status.json').read_text())
                assert status['state'] == 'completed'
                metrics = json.loads((folder/'metrics_test_best_1.json').read_text())
                expected = float(metrics['nll_test_time'])
                cache = OUT/f'{name}.npz'
                save('status.json', dict(state='running', region=region, model=family, seed=seed))
                if cache.exists():
                    with np.load(cache) as values:
                        assert str(values['checkpoint_sha256']) == sha
                        np.testing.assert_array_equal(values['edges'], edges)
                        sums = values['segment_nll'].copy()
                        full = float(values['full_nll'])
                else:
                    model, _ = load_and_prepare_model(checkpoint, device, compile=False)
                    full = time_nll(model, sequence, device)
                    assert abs(full/duration-expected) < 2e-5, (name, full/duration, expected)
                    sums = np.array([time_nll(model, part, device) for part in segmented])
                    assert abs(sums.sum()/duration-expected) < 2e-5, (name, sums.sum()/duration, expected)
                    np.savez_compressed(cache, segment_nll=sums, full_nll=full,
                        edges=edges, checkpoint_sha256=sha)
                    del model
                    torch.cuda.empty_cache()
                assert abs(sums.sum()/duration-expected) < 2e-5
                scores.append(sums)
                fits.append(dict(region=region, model=family, seed=seed, time_NLL_per_day=full/duration,
                    segment_sum_per_day=sums.sum()/duration, saved_metric=expected,
                    reconstruction_error_per_day=sums.sum()/duration-expected))
                checkpoint_records.append(dict(name=name, path=str(checkpoint), sha256=sha))
                pd.DataFrame(fits).to_csv(OUT/'fit_scores.csv', index=False)
                print(name, 'full score and segment reconstruction passed', flush=True)
            arrays[family] = np.stack(scores)
        for baseline in ['RECAST', 'ETAS']:
            for block in [1, 2, 4]:
                rng = np.random.default_rng(20261102)
                draws = []
                for _ in range(2000):
                    starts = rng.integers(0, 20, size=int(np.ceil(20/block)))
                    index = ((starts[:, None]+np.arange(block)) % 20).ravel()[:20]
                    neural_fits = rng.integers(0, 5, size=5)
                    reference_fits = rng.integers(0, len(arrays[baseline]), size=len(arrays[baseline]))
                    neural = arrays['EM-EQF'][np.ix_(neural_fits, index)].sum(axis=1).mean()
                    reference = arrays[baseline][np.ix_(reference_fits, index)].sum(axis=1).mean()
                    draws.append((reference-neural)/duration)
                low, high = np.quantile(draws, [.025, .975])
                intervals.append(dict(region=region, baseline=baseline, neural_fits=5,
                    baseline_fits=len(arrays[baseline]), calendar_segments=20, block_segments=block,
                    bootstrap_replicates=2000, delta_per_day=(arrays[baseline].sum(axis=1).mean()-arrays['EM-EQF'].sum(axis=1).mean())/duration,
                    low95_per_day=low, high95_per_day=high, sign='positive favors EM-EQF', scope=plan['scope']))
        pd.DataFrame(intervals).to_csv(OUT/'paired_time_block_intervals.csv', index=False)
    save('checkpoint_manifest.json', checkpoint_records)
    save('status.json', dict(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
        checked_fits=len(fits), paired_intervals=len(intervals)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        save('status.json', dict(state='failed', error=str(error), failed_utc=datetime.now(timezone.utc).isoformat()))
        raise
