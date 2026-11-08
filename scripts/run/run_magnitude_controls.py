"""Reproduce the matched no-auxiliary-loss GR ablation and paired scoring."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/run'))
sys.path.insert(0, str(ROOT/'scripts/analyze'))
from run_magnitude_models import OUT, load_data, train_fit
from compare_magnitude_models import block_intervals
from src.models.tpp.magnitude_density import MagnitudeDensityHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    statuses = []
    recipe = dict(family='gr', components=1, features='context_history')
    if not args.score_only:
        for variant in ['main', 'sensitivity']:
            for seed in range(5):
                statuses.append(train_fit('scsn', variant, seed, recipe, 'matched_control'))
        pd.DataFrame(statuses).to_csv(OUT/'reports/matched_control_validation_scores.csv', index=False)
    report = OUT/'reports/comparison'
    rows, intervals = [], []
    for region, mode, stage in [('swc', 'context', 'confirm'), ('scsn', 'context_history', 'matched_control')]:
        for variant in ['main', 'sensitivity']:
            values, originals, selected = [], [], []
            for seed in range(5):
                path = OUT/'runs'/f'{stage}_{region}_{variant}_gr1_{mode}_seed_{seed}'/'best_head.pth'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                data = load_data(region, variant, seed, mode)
                mask = data['split']==3
                features = (data['features'][mask]-saved['normalization_mean'])/saved['normalization_std']
                model = MagnitudeDensityHead(saved['input_dim'], 'gr', 1,
                    saved['Mc'], saved['step'], hidden=saved['config']['hidden']).cuda().eval()
                model.load_state_dict(saved['state_dict'])
                with torch.inference_mode():
                    lp = model(torch.tensor(features, device='cuda')).log_prob(
                        torch.tensor(data['magnitude'][mask], device='cuda')).cpu().numpy()
                with np.load(report/f'{region}_{variant}_seed_{seed}.npz') as d:
                    np.testing.assert_array_equal(data['magnitude'][mask], d['magnitude'])
                    selected.append(d['selected_logp'].copy())
                originals.append(data['old_logp'][mask])
                values.append(lp)
                rows.append(dict(region=region, variant=variant, seed=seed,
                    model='GR_no_aux_matched_features', nll_event=float(-lp.mean())))
                np.savez_compressed(report/f'{region}_{variant}_gr_control_seed_{seed}.npz',
                    logp=lp, times=data['times'][mask], magnitude=data['magnitude'][mask])
                del model
            period = json.loads((OUT/'features'/f'{region}_{variant}_events.json').read_text())['periods']['test']
            for name, gains in [('selected_vs_GR_no_aux', np.asarray(selected)-values),
                                ('GR_no_aux_vs_original', np.asarray(values)-originals)]:
                for row in block_intervals(gains, data['times'][mask], period['start'], period['end']):
                    intervals.append(dict(region=region, variant=variant, pair=name, **row))
    pd.DataFrame(rows).to_csv(report/'no_aux_GR_control_scores.csv', index=False)
    pd.DataFrame(intervals).to_csv(report/'no_aux_GR_control_intervals.csv', index=False)
    print(pd.DataFrame(rows).groupby(['region','variant']).nll_event.mean().to_string())
    print(pd.DataFrame(intervals).query('block_segments==4').to_string(index=False))


if __name__=='__main__':
    main()
