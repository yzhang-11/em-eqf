"""Paired magnitude scores: five independent fits plus calendar-block resampling."""
import json
import argparse
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
from src.data import Batch, Catalog
from src.catalogs.pathing import build_tpp_catalog_init_kwargs
from src.train.config_setup import load_and_prepare_model

OUT = ROOT / 'experiments/revision/main/reports/generation_comparison'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['main','sensitivity'],default='main')
    variant=parser.parse_args().variant
    out=OUT if variant=='main' else OUT.with_name('generation_comparison_sensitivity')
    reference_folder='dynamic_b' if variant=='main' else 'dynamic_b_sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for region in ['swc', 'scsn']:
        c = OmegaConf.load(ROOT / 'experiments/revision/main/configs' / f'generation_{region}_{variant}_seed_0.yaml')
        cls = Catalog.by_name(c.dataset + '-Standard')
        cat = cls(**build_tpp_catalog_init_kwargs(cls, ROOT / 'data' / c.dataset, c.catalog_cfg))
        seq = cat.test[0].double()
        duration = seq.t_end - seq.t_nll_start
        test = seq.arrival_times >= seq.t_nll_start
        times, targets = seq.arrival_times[test].numpy(), seq.mag[test].numpy()
        scores = []
        for seed in range(5):
            run = ROOT / 'experiments/revision/main/runs' / f'generation_{region}_{variant}_seed_{seed}'
            status = json.loads((run / 'job_status.json').read_text())
            assert status['state'] == 'completed'
            model, _ = load_and_prepare_model(run / 'best_model_1.pth', torch.device('cuda:0'), compile=False)
            batch = Batch.from_list([seq]).to(torch.device('cuda:0'))
            with torch.no_grad():
                context = model.get_context(batch)
                distribution = model.get_magnitude_dist(context, inter_times=batch.inter_times)
                nll = -distribution.log_prob(batch.mag)[batch.nll_event_mask.bool()]
            actual = nll.cpu().numpy()
            expected = status['metrics']['nll_test_mag']
            assert abs(actual.sum() / duration - expected) < 1e-5
            scores.append(actual)
            del model, batch, context, distribution, nll
            torch.cuda.empty_cache()
        neural = np.asarray(scores)
        references = {}
        for name in ['static_GR', 'causal_discounted_gamma_mean']:
            with np.load(ROOT / 'experiments/revision/main/runs/references' / reference_folder / f'{region}_{name}_test.npz') as reference:
                np.testing.assert_allclose(reference['target'], targets, rtol=0, atol=1e-6)
                np.testing.assert_allclose(reference['time'], times, rtol=0, atol=1e-9)
                references[name] = reference['nll'].copy()
        np.savez_compressed(out / f'{region}_paired_magnitude_nll.npz',
                            neural=neural, target=targets, time=times, **references)
        edges = np.linspace(seq.t_nll_start, seq.t_end, 21)
        segment = np.clip(np.searchsorted(edges, times, side='right') - 1, 0, 19)
        counts = np.bincount(segment, minlength=20)
        neural_sums = np.asarray([np.bincount(segment, weights=score, minlength=20) for score in neural])
        for name, values in references.items():
            ref_sums = np.bincount(segment, weights=values, minlength=20)
            for block in [1, 2, 4]:
                rng = np.random.default_rng(20261102)
                deltas_day, deltas_event = [], []
                for _ in range(2000):
                    starts = rng.integers(0, 20, size=int(np.ceil(20 / block)))
                    index = ((starts[:, None] + np.arange(block)) % 20).reshape(-1)[:20]
                    seeds = rng.integers(0, 5, size=5)
                    difference = ref_sums[index].sum() - neural_sums[seeds][:, index].sum(axis=1).mean()
                    deltas_day.append(difference / duration)
                    deltas_event.append(difference / counts[index].sum())
                delta = values.sum() - neural.sum(axis=1).mean()
                low, high = np.quantile(deltas_day, [.025, .975])
                low_event, high_event = np.quantile(deltas_event, [.025, .975])
                rows.append(dict(region=region, reference=name, seeds=5, test_events=len(times),
                                 calendar_segments=20, block_segments=block, bootstrap_replicates=2000,
                                 delta_per_day=float(delta / duration), low95_per_day=float(low), high95_per_day=float(high),
                                 delta_per_event=float(delta / len(times)), low95_per_event=float(low_event), high95_per_event=float(high_event),
                                 sign='positive favors neural; inference conditional on fixed catalog and validation-selected settings'))
        print(region, 'paired comparison complete', flush=True)
    pd.DataFrame(rows).to_csv(out / 'paired_magnitude_block_intervals.csv', index=False)
    print(pd.DataFrame(rows).query('reference == "causal_discounted_gamma_mean"').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
