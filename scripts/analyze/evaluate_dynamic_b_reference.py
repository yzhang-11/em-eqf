"""Non-neural causal b references using the existing updater and GR density."""
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
from src.data import Catalog
from src.catalogs.pathing import build_tpp_catalog_init_kwargs
from src.models.updaters.bayesian_b_updater import BayesianGRBUpdater
from src.distributions import GutenbergRichter

OUT = ROOT / 'experiments/revision/main/runs/references/dynamic_b'


def causal_b(sequence, mc, config, delta):
    updater = BayesianGRBUpdater(Mc=mc, delta=delta, a0=float(config.a0),
                                init_b_target=float(config.init_b_target), write_back=False)
    post = updater.fit(sequence)
    # Stored updater states include the current mark: shift once for prediction.
    prior_a = torch.cat([sequence.mag.new_tensor([updater.a0]), post['a_t'][:-1]])
    prior_s = torch.cat([sequence.mag.new_tensor([updater.s0]), post['s_t'][:-1]])
    return prior_a / (prior_s * np.log(10.))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['main','sensitivity'],default='main')
    variant=parser.parse_args().variant
    out=OUT if variant=='main' else OUT.with_name('dynamic_b_sensitivity')
    out.mkdir(parents=True, exist_ok=True)
    rows, selections = [], []
    for region in ['swc', 'scsn']:
        c = OmegaConf.load(ROOT / 'experiments/revision/main/configs' / f'generation_{region}_{variant}_seed_0.yaml')
        cls = Catalog.by_name(c.dataset + '-Standard')
        cat = cls(**build_tpp_catalog_init_kwargs(cls, ROOT / 'data' / c.dataset, c.catalog_cfg))
        seq = cat.full_sequence.double()
        mags, times = seq.mag, seq.arrival_times
        val = (times >= cat.val[0].t_nll_start) & (times < cat.val[0].t_end)
        test = (times >= cat.test[0].t_nll_start) & (times < cat.test[0].t_end)
        candidates = []
        for delta in [.95, .98, .995, .999]:
            predicted_b = causal_b(seq, float(c.Mc), c.b_updater_cfg, delta)
            nll = -GutenbergRichter(predicted_b, mag_min=float(c.Mc)).log_prob(mags)
            candidates.append((float(nll[val].mean()), delta, nll, predicted_b))
        _, selected_delta, dynamic_nll, predicted_b = min(candidates, key=lambda item: item[0])
        static_nll = -GutenbergRichter(mags.new_tensor(float(c.richter_b)), mag_min=float(c.Mc)).log_prob(mags)
        selections.append(dict(region=region, selected_delta=selected_delta,
                               rule='minimum validation magnitude NLL; test not used for selection',
                               candidates=[dict(delta=d, validation_nll_per_event=v) for v,d,_,_ in candidates],
                               a0=float(c.b_updater_cfg.a0), init_b_target=float(c.b_updater_cfg.init_b_target)))
        for model, values in [('static_GR', static_nll), ('causal_discounted_gamma_mean', dynamic_nll)]:
            for split, mask, interval in [('validation', val, cat.val[0]), ('test', test, cat.test[0])]:
                n = int(mask.sum())
                duration = interval.t_end - interval.t_nll_start
                score = values[mask]
                assert torch.isfinite(score).all()
                rows.append(dict(region=region, model=model, split=split, Mc=float(c.Mc), events=n,
                                 duration_days=duration, magnitude_nll_per_event=float(score.mean()),
                                 magnitude_nll_per_day=float(score.sum()/duration)))
            np.savez_compressed(out / f'{region}_{model}_test.npz',
                                target=mags[test].numpy(), nll=values[test].numpy(),
                                time=times[test].numpy(), b=predicted_b[test].numpy() if model.startswith('causal') else float(c.richter_b))
        # A changed current observation cannot change its own predicted b.
        altered = seq.__class__(inter_times=seq.inter_times.clone(), mag=seq.mag.clone())
        changed_index = int(torch.nonzero(test)[0])
        altered.mag[changed_index] += .5
        altered_b = causal_b(altered, float(c.Mc), c.b_updater_cfg, selected_delta)
        assert altered_b[changed_index] == predicted_b[changed_index]
        assert altered_b[changed_index + 1] != predicted_b[changed_index + 1]
        print(region, 'causality and scoring passed', flush=True)
    pd.DataFrame(rows).to_csv(out / 'magnitude_reference_scores.csv', index=False)
    (out / 'validation_selection.json').write_text(json.dumps(selections, indent=2))
    print(pd.DataFrame(rows).query('split == "test"').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
