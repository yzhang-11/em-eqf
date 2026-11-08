"""Complete the remaining prespecified partial-versus-scratch comparison."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.utils.bootstrap_ci import BootstrapConfig
from src.utils.classifier_compare_utils import paired_metric_bootstrap

BASE=ROOT/'experiments/revision/controls'


def main():
    rows=[]
    for task in ['clf','reg']:
        predictions={};target=None
        for variant in ['partial','scratch']:
            values=[]
            for seed in range(5):
                folder=BASE/'runs'/f'transfer_{task}_{variant}_seed_{seed}'
                with np.load(folder/'predictions_test.npz') as file:
                    y=file['target'].reshape(-1)
                    if target is None:
                        target=y
                    np.testing.assert_allclose(y,target,rtol=0,atol=1e-5)
                    values.append(expit(file['logits'].reshape(-1)) if task=='clf' else file['prediction'].reshape(-1))
            predictions[variant]=np.asarray(values)
        for metric in (['ap','auc'] if task=='clf' else ['rmse','mae']):
            for block in [4,9,18,36]:
                result=paired_metric_bootstrap(y_true=target,metric_name=metric,
                    task='classification' if task=='clf' else 'regression',
                    metric_config=BootstrapConfig(n_resamples=2000,seed=20261102,sampling='block',block_size=block),
                    use_hierarchical_seed_sample=True,seed_aggregation='mean',point_estimate_mode='mean',
                    model_seed_predictions=predictions)
                delta=result.pairwise_deltas[('partial','scratch')]
                rows.append(dict(task=task,metric=metric,block_windows=block,delta=delta.point_estimate,
                    low95=delta.ci_low,high95=delta.ci_high,valid_resamples=delta.valid_resamples,
                    total_resamples=delta.total_resamples,sign='partial minus scratch',
                    scope='five paired target-fit seeds; source checkpoint fixed; no familywise multiplicity adjustment'))
                pd.DataFrame(rows).to_csv(BASE/'reports/partial_vs_scratch_block_intervals.csv',index=False)
                print(task,metric,block,rows[-1],flush=True)


if __name__=='__main__':
    main()
