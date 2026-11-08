"""Reuse existing hierarchical paired bootstrap for threshold-free metrics."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
from src.utils.bootstrap_ci import BootstrapConfig
from src.utils.classifier_compare_utils import paired_metric_bootstrap

BASE = ROOT / 'experiments/revision/main'
OUT = BASE / 'reports/supervised/classification_paired_block_intervals.csv'


def main():
    rows=[]
    for horizon in [10,20,30,60,90]:
        predictions={};target=None
        for model in ['azdx','scratch']:
            values=[]
            for seed in range(5):
                with np.load(BASE/'runs'/f'clf_{model}_h{horizon}_seed_{seed}'/'predictions_test.npz') as file:
                    y=file['target'].reshape(-1).astype(int)
                    if target is None:target=y
                    np.testing.assert_array_equal(y,target)
                    values.append(expit(file['logits'].reshape(-1)))
            predictions[model]=np.asarray(values)
        for metric in ['ap','auc']:
            for block in [4,9,18,36]:
                result=paired_metric_bootstrap(y_true=target,metric_name=metric,
                    metric_config=BootstrapConfig(n_resamples=2000,seed=20261102,sampling='block',block_size=block),
                    use_hierarchical_seed_sample=True,seed_aggregation='mean',point_estimate_mode='mean',
                    model_seed_predictions=predictions)
                delta=result.pairwise_deltas[('azdx','scratch')]
                rows.append(dict(horizon=horizon,metric=metric,block_windows=block,block_shift_days=block*10,
                    delta=delta.point_estimate,low95=delta.ci_low,high95=delta.ci_high,
                    valid_resamples=delta.valid_resamples,total_resamples=delta.total_resamples,
                    skipped_fraction=1-delta.valid_resamples/delta.total_resamples,
                    stable_class_support=delta.valid_resamples/delta.total_resamples>=.95,
                    note='positive favors pretraining; shared paired model seeds; interval conditional on resamples containing both classes'))
                pd.DataFrame(rows).to_csv(OUT,index=False)
                print(horizon,metric,block,'valid',delta.valid_resamples,flush=True)


if __name__=='__main__':
    main()
