"""Freeze causal contexts for magnitude-head comparisons, without fitting on test data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import lfilter
import torch
from omegaconf import OmegaConf

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
from src.data import Batch, TppDataset
from src.train.config_setup import load_and_prepare_model

OUT = ROOT / 'experiments/revision/magnitude_models'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def history_features(mag, intervals, mc):
    """Features at i contain observations strictly before i."""
    values = np.column_stack([mag - mc, (mag - mc) ** 2,
                              mag >= mc + .5, mag >= mc + 1., mag >= mc + 2.,
                              np.log1p(intervals)])
    features = []
    for length in [32, 128, 512]:
        alpha = 1. / length
        filtered = lfilter([alpha], [1., -(1. - alpha)], values, axis=0)
        weight = 1. - (1. - alpha) ** np.arange(1, len(mag) + 1)
        filtered /= weight[:, None]
        features.append(np.vstack([np.zeros((1, values.shape[1])), filtered[:-1]]))
    return np.concatenate(features, axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--variants', nargs='+', default=['main', 'sensitivity'])
    args = parser.parse_args()
    for folder in ['configs', 'runs', 'reports', 'logs', 'features']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    periods = pd.read_csv(ROOT / 'experiments/revision/generation_baselines/reports/generation_periods.csv')
    records = []
    device = torch.device('cuda:0')
    for region in ['swc', 'scsn']:
        for variant in args.variants:
            config_path = ROOT / 'experiments/revision/main/configs' / f'generation_{region}_{variant}_seed_0.yaml'
            cfg = OmegaConf.load(config_path)
            mc = float(cfg.Mc)
            step = .1 if region == 'swc' else .01
            matches = [p for p in Path(cfg.catalog_cfg.root_dir).glob('*/catalog_cfg.json')
                       if json.loads(p.read_text()).get('time_processing') == 'stable-half-resolution-f64-v1'
                       and json.loads(p.read_text())['mag_completeness'] == mc]
            assert len(matches) == 1
            sequence_path = matches[0].parent / 'full_sequence.pt'
            sequence = TppDataset.load_from_disk(sequence_path)[0]
            magnitude = sequence.mag.cpu().numpy().copy()
            times = sequence.arrival_times.cpu().numpy().copy()
            interval = sequence.inter_times.cpu().numpy()[:-1].copy()
            subset = periods[(periods.region == region) & (periods.variant == variant)]
            split = np.zeros(len(magnitude), dtype=np.int8)
            metadata = {}
            for index, name in enumerate(['train', 'validation', 'test'], 1):
                row = subset[subset.period == name].iloc[0]
                start, end = float(row.relative_start), float(row.relative_end_exclusive)
                mask = (times >= start) & (times < end)
                assert int(mask.sum()) == int(row.events)
                split[mask] = index
                metadata[name] = dict(start=start, end=end, events=int(mask.sum()), days=end-start)
            bins = np.rint((magnitude.astype(np.float64) - mc) / step).astype(np.int64)
            assert np.max(abs(magnitude - (mc + bins * step))) < 2e-6
            assert magnitude.max() < 10
            history = history_features(magnitude.astype(float), interval, mc)
            data_path = OUT / 'features' / f'{region}_{variant}_events.npz'
            if not data_path.exists():
                np.savez_compressed(data_path, magnitude=magnitude, times=times, bins=bins,
                                    split=split, history=history, Mc=mc, step=step)
            save(data_path.with_suffix('.json'), dict(region=region, variant=variant, Mc=mc, step=step,
                 grid='reported grid value Mc+k*step; all bins [Mc+k*step,Mc+(k+1)*step), common upper bound 10',
                 sequence_path=str(sequence_path), sequence_sha256=digest(sequence_path),
                 source_path=str(cfg.catalog_cfg.catalog_file), source_sha256=digest(Path(cfg.catalog_cfg.catalog_file)),
                 periods=metadata, history='strictly prior marks/intervals; no target mark or target interval'))
            sequence = sequence.double()
            sequence.t_nll_start = metadata['test']['start']
            batch = Batch.from_list([sequence]).to(device)
            for seed in args.seeds:
                run = ROOT / 'experiments/revision/main/runs' / f'generation_{region}_{variant}_seed_{seed}'
                checkpoint = run / 'best_model_1.pth'
                destination = OUT / 'features' / f'{region}_{variant}_seed_{seed}.npz'
                if destination.exists():
                    info = json.loads(destination.with_suffix('.json').read_text())
                    assert info['checkpoint_sha256'] == digest(checkpoint)
                    records.append(info)
                    continue
                save(OUT / 'reports/features_status.json', dict(state='running', region=region, variant=variant, seed=seed))
                model, _ = load_and_prepare_model(checkpoint, device, compile=False)
                assert model.b_filter is None
                with torch.inference_mode():
                    context = model.get_context(batch)
                    distribution, b = model.get_magnitude_dist(context, return_b=True)
                    old_nll = -distribution.log_likelihood(batch.mag, batch.nll_event_mask.bool())
                    per_day = float(old_nll / metadata['test']['days'])
                    expected = json.loads((run / 'metrics_test_best_1.json').read_text())['nll_test_mag']
                    assert abs(per_day - expected) < 1e-5, (region, variant, seed, per_day, expected)
                    context_array = context[0, :len(magnitude)].cpu().numpy().copy()
                    b_array = b[0, :len(magnitude)].double().cpu().numpy().copy()
                rate = b_array * np.log(10.)
                lower = bins * step
                upper = np.minimum(lower + step, 10. - mc)
                logp = -rate*lower + np.log(-np.expm1(-rate*(upper-lower))) - np.log(-np.expm1(-rate*(10.-mc)))
                assert np.isfinite(logp).all()
                np.savez_compressed(destination, context=context_array, old_b=b_array, old_logp=logp)
                info = dict(region=region, variant=variant, seed=seed, checkpoint=str(checkpoint),
                            checkpoint_sha256=digest(checkpoint), old_mag_nll_day=per_day,
                            saved_mag_nll_day=expected, absolute_error=abs(per_day-expected),
                            feature_file=str(destination), feature_sha256=digest(destination))
                save(destination.with_suffix('.json'), info)
                records.append(info)
                print(json.dumps(info), flush=True)
                del model, context, distribution, b
                torch.cuda.empty_cache()
            del batch, sequence
    save(OUT / 'reports/features_status.json', dict(state='completed', fits=len(records)))
    save(OUT / 'features/manifest.json', dict(fits=records, extractor_sha256=digest(Path(__file__))))


if __name__ == '__main__':
    main()
