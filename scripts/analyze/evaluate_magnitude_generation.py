"""Autoregressive window checks for frozen validation-selected magnitude heads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
from src.data import TppDataset
from src.models.tpp.magnitude_model import load_magnitude_model
from src.utils.catalog_tests import compute_magnitude_distribution, magnitude_test_from_counts

OUT = ROOT/'experiments/revision/magnitude_models'
REFERENCE = ROOT/'experiments/revision/main/reports'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fair_crps(observed, samples):
    """Unbiased finite-ensemble estimate, in event-count units."""
    values = np.sort(np.asarray(samples, dtype=float))
    n = len(values)
    return float(np.abs(values-observed).mean() -
                 ((2*np.arange(1, n+1)-n-1)*values).sum()/(n*(n-1)))


def score(region, window, a, b, model, counts, maxima, histogram, observed, mc, step):
    observed_histogram = compute_magnitude_distribution(observed, mc)[1]
    observed_count = len(observed)
    result = magnitude_test_from_counts(histogram, observed_histogram, get_counts=np.asarray,
        verbose=False, debug=False, return_quantiles=True)
    valid = result['obs_d_statistic'] is not None
    row = dict(region=region, seed=0, window=window, start=a, end=b,
        partial_window=b-a<30-1e-8, model=model, draws=len(counts),
        observed_count=observed_count, predicted_count=float(counts.mean()),
        count_CRPS=fair_crps(observed_count, counts),
        M_test_valid=valid, M_test_reject_05=result['delta_1']<.05 if valid else None,
        maximum_sim_count=int(counts.max()))
    # Common reported-grid observation rule. Old continuous maxima are floored;
    # new grid values are rounded to their exact integer label before scoring.
    if model == 'original':
        state = np.floor((np.nan_to_num(maxima, nan=mc-step)-mc)/step+1e-7)
    else:
        state = np.rint((np.nan_to_num(maxima, nan=mc-step)-mc)/step)
    state = np.where(counts>0, state, -1)
    target = round((float(observed.max())-mc)/step) if observed_count else -1
    for level in [.8, .9, .95]:
        alpha = (1-level)/2
        low, high = np.quantile(counts, [alpha, 1-alpha], method='inverted_cdf')
        row[f'count_coverage_{int(level*100)}'] = bool(low<=observed_count<=high)
        low, high = np.quantile(state, [alpha, 1-alpha], method='inverted_cdf')
        row[f'maximum_state_coverage_{int(level*100)}'] = bool(low<=target<=high)
    for threshold in [mc+.5, mc+1., 5., 6.]:
        if threshold<=mc:
            continue
        probability = float(np.mean(state>=round((threshold-mc)/step)))
        row[f'maximum_Brier_{threshold:g}'] = (probability-(target>=round((threshold-mc)/step)))**2
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-window-only', action='store_true')
    parser.add_argument('--dynamics', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = OUT/('reports/dynamics/generation' if args.dynamics else 'reports/generation')
    folder.mkdir(exist_ok=True)
    selection_path = OUT/('reports/dynamics/selection.json' if args.dynamics else 'selection.json')
    selection = json.loads(selection_path.read_text())
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=digest(selection_path), fits=[0], draws_per_window=128,
        horizon_days=30, scope='all original test windows; last partial window reported separately',
        original='first 128 of frozen original 512 simulations, with checkpoint/hash verification',
        selected='new autoregressive simulation; all observed pre-origin history, independent trajectory updates',
        common_observation='original latent maximum floored to catalog step; new grid maximum rounded to exact label',
        M_test='same PyCSEP-equivalent helper, .1 bins, observed float32 precision preserved',
        uncertainty='single fitted seed and finite Monte Carlo ensemble; diagnostic, not a five-fit superiority claim',
        selection_unchanged=True, first_window_only=args.first_window_only)
    path = folder/'plan.json'
    if path.exists():
        previous = json.loads(path.read_text())
        assert all(previous[k]==v for k,v in plan.items() if k not in {'created_utc','first_window_only'})
        if previous['first_window_only'] and not args.first_window_only:
            previous['first_window_only'] = False
            save(path, previous)
    else:
        save(path, plan)
    rows = []
    for region_index, (region, cache) in enumerate([
            ('swc', 'ChuanDian/catalogs/c6afef21'), ('scsn', 'SCEDC/catalogs/25a12469')]):
        if args.dynamics:
            row = next(r for r in selection['best'] if r['region']==region and r['variant']=='main'
                       and r['family']==selection['choices'][region])
            head = Path(row['checkpoint'])
        else:
            recipe = selection['recipes'][region]['selected']
            name = f"confirm_{region}_main_{recipe['family']}{recipe['components']}_{recipe['features']}_seed_0"
            head = OUT/'runs'/name/'best_head.pth'
        head_hash = digest(head)
        if args.dynamics:
            from src.models.tpp.magnitude_dynamics_model import load_gas_magnitude_model
            model, _ = load_gas_magnitude_model(head, torch.device('cuda:0'))
        else:
            model, _ = load_magnitude_model(head, torch.device('cuda:0'))
        original_hash = digest(ROOT/f'experiments/revision/main/runs/generation_{region}_main_seed_0/best_model_1.pth')
        original = TppDataset.load_from_disk(ROOT/'data'/cache/'full_sequence.pt')[0]
        sequence = original.double()
        sequence.t_nll_start = sequence.t_start
        times = sequence.arrival_times.numpy()
        magnitudes = original.mag.float().numpy()
        mc, step = model.density_head.mag_min, model.density_head.step
        with np.load(REFERENCE/'calibration'/f'{region}_seed_0.npz') as d:
            assert str(d['checkpoint_sha256'])==original_hash
            starts, ends = d['start'].copy(), d['end'].copy()
        for window, (a,b) in enumerate(zip(starts, ends)):
            if args.first_window_only and window:
                break
            began = time.monotonic()
            observed = magnitudes[(times>=a)&(times<b)]
            with np.load(REFERENCE/'calibration/windows'/f'{region}_seed_0'/f'{window:04d}.npz') as d:
                assert str(d['checkpoint_sha256'])==original_hash
                assert int(d['observed_count'])==len(observed)
                assert float(d['start'])==a and float(d['end'])==b
                old_count, old_maximum = d['sim_count'][:128].copy(), d['sim_maximum'][:128].copy()
                sampling_seed = int(d['sampling_seed'])
            with np.load(REFERENCE/'magnitude_test/windows'/f'{region}_seed_0'/f'{window:04d}.npz') as d:
                assert str(d['checkpoint_sha256'])==original_hash and int(d['sampling_seed'])==sampling_seed
                old_histogram = d['simulation_histograms'][:128].copy()
                np.testing.assert_array_equal(old_histogram.sum(-1), old_count)
            window_folder = folder/'windows'/region
            window_folder.mkdir(parents=True, exist_ok=True)
            path = window_folder/f'{window:04d}.npz'
            if path.exists():
                with np.load(path) as d:
                    assert str(d['head_sha256'])==head_hash and int(d['sampling_seed'])==sampling_seed
                    counts, maxima, histogram = d['sim_count'].copy(), d['sim_maximum'].copy(), d['simulation_histograms'].copy()
                    assert float(d['start'])==a and float(d['end'])==b
            else:
                past = sequence.get_subsequence(sequence.t_start, a).to('cuda:0')
                past.t_nll_start = past.t_start
                torch.manual_seed(sampling_seed)
                np.random.seed(sampling_seed)
                with torch.inference_mode():
                    forecasts = model.sample(batch_size=128, duration=float(b-a), past_seq=past,
                        max_sample_len=16384, return_sequences=True)
                counts = np.array([len(x) for x in forecasts])
                maxima = np.array([float(x.mag.max()) if len(x) else np.nan for x in forecasts])
                histogram = np.stack([compute_magnitude_distribution(x.mag.float().cpu().numpy(), mc)[1] for x in forecasts])
                np.testing.assert_array_equal(histogram.sum(-1), counts)
                assert np.array_equal(np.isnan(maxima), counts==0)
                np.savez_compressed(path, start=a, end=b, sim_count=counts, sim_maximum=maxima,
                    simulation_histograms=histogram, observed_histogram=compute_magnitude_distribution(observed, mc)[1],
                    head_sha256=head_hash, base_checkpoint_sha256=original_hash, sampling_seed=sampling_seed)
                del forecasts, past
            for label, c, m, h in [('original', old_count, old_maximum, old_histogram),
                                    ('selected', counts, maxima, histogram)]:
                rows.append(score(region, window, float(a), float(b), label, c, m, h, observed, mc, step))
            frame = pd.DataFrame(rows)
            frame.to_csv(folder/'window_scores.csv', index=False)
            save(folder/'status.json', dict(state='running', region=region, window=window,
                 completed_fit_windows=len(frame)//2, elapsed_seconds=time.monotonic()-began))
            print(region, window, len(starts), 'seconds', round(time.monotonic()-began, 2),
                  'max_count', int(counts.max()), flush=True)
        del model, sequence
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    metrics = ['count_CRPS', 'M_test_reject_05']+[x for x in frame if '_coverage_' in x or 'maximum_Brier_' in x]
    summary = frame.groupby(['region','model','partial_window']).agg(windows=('window','count'),
        **{key:(key,'mean') for key in metrics}).reset_index()
    summary.to_csv(folder/'summary.csv', index=False)
    save(folder/'status.json', dict(state='pilot_completed' if args.first_window_only else 'completed',
        completed_utc=datetime.now(timezone.utc).isoformat(), fit_windows=len(frame)//2,
        independent_fits_per_region=1, draws_per_window=128))
    print(summary.to_string(index=False), flush=True)


if __name__=='__main__':
    main()
