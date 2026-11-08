"""Replay frozen calibration draws, retaining histograms for the existing M-test."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/analyze'))
from evaluate_full_window_calibration import HistorySampler
from src.catalogs.pathing import build_tpp_catalog_init_kwargs
from src.data import Catalog
from src.train.config_setup import load_and_prepare_model
from src.utils.catalog_tests import compute_magnitude_distribution, magnitude_test_from_counts
from src.utils.forecast_eval import (
    SlidingWindowEvaluationRange, SlidingWindowForecastConfig, run_sliding_window_forecast,
)

BASE = ROOT/'experiments/revision/main'
OUT = BASE/'reports/magnitude_test'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, payload):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temp.replace(path)


class HistogramCatalog:
    def __init__(self, counts):
        self.counts = np.asarray(counts)
        self.event_count = self.counts.sum()
        self.name = 'histogram catalog'

    def magnitude_counts(self):
        return self.counts


def validate_existing_method():
    """Compare the repository helper with installed PyCSEP, including empty draws."""
    from csep.core.catalog_evaluations import magnitude_test
    rng = np.random.default_rng(20261102)
    simulations = rng.poisson([9, 4, 2, 1, .1], size=(512, 5))
    simulations[:7] = 0
    observed = np.array([8, 5, 1, 0, 1])
    cats = [HistogramCatalog(x) for x in simulations]

    class Forecast(list):
        region = SimpleNamespace(magnitudes=np.arange(5))
        expected_rates = HistogramCatalog(simulations.mean(axis=0))
        min_magnitude = 3.0
        name = 'validation ensemble'

    reference = magnitude_test(Forecast(cats), HistogramCatalog(observed), verbose=False)
    result = magnitude_test_from_counts(simulations, observed, get_counts=np.asarray,
        verbose=False, debug=False, return_quantiles=True)
    np.testing.assert_array_equal(reference.test_distribution, result['test_distribution'])
    assert reference.observed_statistic == result['obs_d_statistic']
    np.testing.assert_array_equal(reference.quantile, [result['delta_1'], result['delta_2']])
    assert len(result['test_distribution']) == 505
    empty = magnitude_test_from_counts(simulations, np.zeros(5), get_counts=np.asarray,
        verbose=False, debug=False, return_quantiles=True)
    assert empty['obs_d_statistic'] is None and not empty['test_distribution']
    # Preserve float32 input precision: the existing PyCSEP binning routine
    # accounts for serialized roundoff at lower bin edges.
    bins, hist = compute_magnitude_distribution(np.array([3.0, 3.1, 3.2], dtype=np.float32), 3.0)
    np.testing.assert_array_equal(hist[:3], [1, 1, 1])
    return dict(state='passed', catalog_draws=512, nonempty_draws=505,
        installed_pycsep_equivalence='exact statistic, distribution and quantiles',
        empty_observation='invalid M-test, retained in denominator accounting',
        bin_edge_roundoff='float32 boundary case passed',
        pycsep_method_sha256=hashlib.sha256(inspect.getsource(magnitude_test).encode()).hexdigest(),
        primary_reference='https://github.com/SCECcode/pycsep/blob/main/csep/core/catalog_evaluations.py')


def histogram(mag, mc):
    arr = mag.detach().cpu().numpy() if torch.is_tensor(mag) else np.asarray(mag)
    assert np.isfinite(arr).all()
    assert not len(arr) or (arr.min() >= mc-1e-6 and arr.max() <= 10+1e-6)
    bins, values = compute_magnitude_distribution(arr, mc, max_mw=10, dmw=.1)
    assert values.sum() == len(arr) and (values >= 0).all()
    return bins, values.astype(np.int32)


class HistogramSampler(HistorySampler):
    def __init__(self, model, mc):
        super().__init__(model)
        self.mc = mc
        self.histograms = None

    def sample(self, **kwargs):
        forecasts = super().sample(**kwargs)
        self.histograms = np.stack([histogram(fc.mag, self.mc)[1] for fc in forecasts])
        np.testing.assert_array_equal(self.histograms.sum(axis=1), [len(fc) for fc in forecasts])
        return forecasts


def summarize(frame):
    rows = []
    for (region, seed), group in frame.groupby(['region', 'seed']):
        for scope, part in [('all_windows', group), ('full_30day', group[~group.partial_window]),
                            ('final_partial', group[group.partial_window])]:
            valid = part[part.valid]
            rows.append(dict(region=region, seed=int(seed), scope=scope, windows=len(part),
                valid_windows=len(valid), invalid_windows=len(part)-len(valid),
                reject_fraction=float(valid.reject_05.mean()) if len(valid) else None,
                mean_nonempty_simulations=float(part.nonempty_simulations.mean())))
    pd.DataFrame(rows).to_csv(OUT/'fit_scores.csv', index=False)
    intervals = []
    rng = np.random.default_rng(20261102)
    for region, group in frame[~frame.partial_window].groupby('region'):
        table = group.pivot(index='seed', columns='window', values='reject_05').to_numpy(dtype=float)
        assert table.shape[0] == 5
        for block in [1, 2, 4, 8]:
            values = []
            for _ in range(2000):
                fits = rng.integers(0, 5, size=5)
                starts = rng.integers(0, table.shape[1], size=int(np.ceil(table.shape[1]/block)))
                indices = ((starts[:, None]+np.arange(block)) % table.shape[1]).ravel()[:table.shape[1]]
                sample = table[np.ix_(fits, indices)]
                if np.isfinite(sample).any():
                    values.append(float(np.nanmean(sample)))
            low, high = np.quantile(values, [.025, .975])
            intervals.append(dict(region=region, block_windows=block, resamples=len(values),
                unique_windows=table.shape[1], fitted_models=5,
                valid_fit_windows=int(np.isfinite(table).sum()),
                rejection_fraction=float(np.nanmean(table)), low95=low, high95=high,
                scope='fitted seeds and common calendar blocks; conditional on fixed catalogs/settings'))
    pd.DataFrame(intervals).to_csv(OUT/'rejection_block_intervals.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wait-followups', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT/'method_validation.json', validate_existing_method())
    plan = dict(windows='same complete test grids as calibration, including final partial windows',
        fitted_seeds=list(range(5)), draws_per_window=512, bin_width=.1, maximum_magnitude=10,
        binning='existing compute_magnitude_distribution / PyCSEP bin1d_vec; preserve input dtype tolerance',
        history='all observed events before origin; existing HistorySampler unchanged',
        random_seed='same 20261102 + 100000*region_index + 1000*fit_seed + window_index as calibration',
        replay_check='exact per-trajectory counts and float32 maxima must match saved calibration draws',
        method='existing magnitude_test_from_counts, checked against installed PyCSEP magnitude_test',
        empty_simulations='retained in forecast union, excluded from normalized M-test null distribution, recorded',
        empty_observation='M-test invalid, retained as such; N-test remains applicable',
        decision='upper discrepancy tail delta_1 < 0.05; inclusive empirical tail; no multiplicity correction',
        aggregate_uncertainty='5 fits and common calendar blocks 1/2/4/8 windows, 2000 resamples',
        primary_reference='https://github.com/SCECcode/pycsep/blob/main/csep/core/catalog_evaluations.py')
    plan_path = OUT/'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        save(plan_path, plan)
    original = json.loads((ROOT/'experiments/revision/generation_baselines/frozen_manifest.json').read_text())
    paths = [Path(x['path']) for key in ['source_files', 'raw_sources'] for x in original[key]]
    paths += [BASE/'configs'/f'generation_{region}_main_seed_0.yaml' for region in ['swc', 'scsn']]
    paths += [Path(__file__), ROOT/'scripts/analyze/evaluate_full_window_calibration.py', BASE/'reports/calibration/plan.json']
    paths += [BASE/'runs'/f'generation_{region}_main_seed_{seed}'/'best_model_1.pth'
              for region in ['swc', 'scsn'] for seed in range(5)]
    manifest = dict(inputs=[dict(path=str(p), sha256=digest(p)) for p in paths], plan_sha256=digest(plan_path))
    frozen_path = OUT/'frozen_manifest.json'
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text()) == manifest
    else:
        save(frozen_path, manifest)
    def verify():
        assert digest(plan_path) == manifest['plan_sha256']
        for entry in manifest['inputs']:
            assert digest(entry['path']) == entry['sha256'], entry['path']
    verify()
    if args.prepare_only:
        save(OUT/'status.json', dict(state='prepared', cpu_method_check='passed', gpu_replay='pending'))
        print('Prepared; installed PyCSEP equivalence passed; GPU replay still pending', flush=True)
        return
    if args.wait_followups:
        save(OUT/'status.json', dict(state='waiting_for_generation_followups', gpu_replay='pending'))
        while True:
            path = ROOT/'experiments/revision/generation_baselines/queue_status.json'
            try:
                state = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(1)
                continue
            if state['state'] == 'failed':
                raise RuntimeError('Follow-up queue failed; holding M-test GPU replay for inspection')
            if state['state'] == 'completed':
                break
            time.sleep(30)
    verify()
    rows = []
    for region_index, region in enumerate(['swc', 'scsn']):
        cfg = OmegaConf.load(BASE/'configs'/f'generation_{region}_main_seed_0.yaml')
        cls = Catalog.by_name(cfg.dataset+'-Standard')
        cat = cls(**build_tpp_catalog_init_kwargs(cls, ROOT/'data'/cfg.dataset, cfg.catalog_cfg))
        seq = cat.test[0].double()
        mc = float(cfg.catalog_cfg.mag_completeness)
        times, magnitudes = seq.arrival_times.numpy(), seq.mag.numpy()
        for seed in range(5):
            verify()
            checkpoint = BASE/'runs'/f'generation_{region}_main_seed_{seed}'/'best_model_1.pth'
            checkpoint_hash = digest(checkpoint)
            calibration = np.load(BASE/'reports/calibration'/f'{region}_seed_{seed}.npz')
            assert str(calibration['checkpoint_sha256']) == checkpoint_hash
            model = None
            folder = OUT/'windows'/f'{region}_seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            for window, (start, end) in enumerate(zip(calibration['start'], calibration['end'])):
                save(OUT/'status.json', dict(state='running', region=region, seed=seed, window=window,
                    total_windows=len(calibration['start']), completed_fit_windows=len(rows)))
                sampling_seed = 20261102+100000*region_index+1000*seed+window
                cache = folder/f'{window:04d}.npz'
                left, right = np.searchsorted(times, [start, end], side='left')
                bins, observed = histogram(magnitudes[left:right], mc)
                assert observed.sum() == calibration['observed_count'][window]
                if cache.exists():
                    with np.load(cache) as data:
                        assert str(data['checkpoint_sha256']) == checkpoint_hash
                        assert int(data['sampling_seed']) == sampling_seed
                        simulations = data['simulation_histograms'].copy()
                        np.testing.assert_array_equal(data['observed_histogram'], observed)
                        np.testing.assert_array_equal(data['magnitude_bins'], bins)
                else:
                    if model is None:
                        model, _ = load_and_prepare_model(checkpoint, torch.device('cuda:0'), compile=False)
                    adapter = HistogramSampler(model, mc)
                    torch.manual_seed(sampling_seed)
                    np.random.seed(sampling_seed)
                    result = run_sliding_window_forecast(adapter, seq, torch.device('cuda:0'),
                        config=SlidingWindowForecastConfig(duration=30, slide_step=30, samples_per_batch=512,
                            evaluation_range=SlidingWindowEvaluationRange(name='test',start=start,end=end,grid_anchor=start),
                            include_truncated_final_window=True, return_sim_count_matrix=True,
                            compute_mag_max=True, return_sim_mag_max_matrix=True,
                            precomputed_counts=calibration['observed_count'][window:window+1],
                            precomputed_mag_max=calibration['observed_maximum'][window:window+1]))
                    np.testing.assert_array_equal(result.sim_count_matrix[0], calibration['sim_count'][window])
                    np.testing.assert_array_equal(result.sim_mag_max_matrix[0], calibration['sim_maximum'][window])
                    simulations = adapter.histograms
                    np.savez_compressed(cache, start=start, end=end, magnitude_bins=bins,
                        observed_histogram=observed, simulation_histograms=simulations,
                        checkpoint_sha256=checkpoint_hash, sampling_seed=sampling_seed,
                        exact_calibration_replay=True)
                    del result, adapter
                assert simulations.shape == (512, len(bins))
                np.testing.assert_array_equal(simulations.sum(axis=1), calibration['sim_count'][window])
                result = magnitude_test_from_counts(simulations, observed, get_counts=np.asarray,
                    verbose=False, debug=False, return_quantiles=True)
                valid = result['obs_d_statistic'] is not None and len(result['test_distribution']) > 0
                td = np.asarray(result['test_distribution'])
                if valid:
                    assert np.isfinite(td).all() and np.isfinite(result['obs_d_statistic'])
                row = dict(region=region, seed=seed, window=window, start=start, end=end,
                    duration=end-start, partial_window=bool(end-start < 30-1e-8), observed_count=int(observed.sum()),
                    simulated_catalogs=512, nonempty_simulations=int((simulations.sum(axis=1)>0).sum()),
                    valid=valid, observed_statistic=result['obs_d_statistic'], upper_tail=result['delta_1'],
                    lower_tail=result['delta_2'], reject_05=bool(result['delta_1'] < .05) if valid else np.nan,
                    exact_calibration_replay=True)
                np.savez_compressed(folder/f'{window:04d}_statistic.npz', null_statistics=td,
                    scaled_union_histogram=result['scaled_union_histogram'])
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT/'window_scores.csv', index=False)
                print(region, seed, window, 'exact replay and M-test completed', flush=True)
            if model is not None:
                del model
                torch.cuda.empty_cache()
            calibration.close()
    summarize(pd.DataFrame(rows))
    save(OUT/'status.json', dict(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
        fitted_models=10, fit_windows=len(rows), unique_windows=len({(r['region'],r['window']) for r in rows}),
        exact_calibration_replay=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        save(OUT/'status.json', dict(state='failed', error=str(error), failed_utc=datetime.now(timezone.utc).isoformat()))
        raise
