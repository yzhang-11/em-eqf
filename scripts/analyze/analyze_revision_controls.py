"""Audit all prespecified control seeds, then reuse paired block bootstrap."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
from src.utils.bootstrap_ci import BootstrapConfig
from src.utils.classifier_compare_utils import paired_metric_bootstrap

BASE = ROOT/'experiments/revision/main'
CONTROL = ROOT/'experiments/revision/controls'
OUT = CONTROL/'reports'


def load(name, parent, task):
    folder = parent/'runs'/name
    status = json.loads((folder/'job_status.json').read_text())
    assert status['state'] == 'completed', name
    with np.load(folder/'predictions_test.npz') as file:
        y = file['target'].reshape(-1).astype(float)
        prediction = expit(file['logits'].reshape(-1)) if task == 'classification' else file['prediction'].reshape(-1)
        if task == 'classification':
            checkpoint = torch.load(folder/'best_model_1.pth', map_location='cpu', weights_only=False)
            assert abs(float(file['threshold'])-checkpoint['val_metrics']['threshold']) < 1e-12
    assert np.isfinite(y).all() and np.isfinite(prediction).all()
    return y, prediction


def write_scores(rows):
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'seed_scores.csv',index=False)
    summary=[]
    for keys,part in frame.groupby(['group','task','variant','scope']):
        row=dict(zip(['group','task','variant','scope'],keys),n=int(part.n.iloc[0]),seeds=len(part))
        for metric in ['RMSE','MAE','AP','AUC','fraction_error_above_1p8']:
            if metric in part:
                row[metric+'_mean']=part[metric].mean()
                row[metric+'_sd']=part[metric].std(ddof=1)
        summary.append(row)
    result=pd.DataFrame(summary)
    result.to_csv(OUT/'summary.csv',index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    plan = json.loads((CONTROL/'plan.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    groups = []
    for family in ['reg_azdx','reg_scratch','lstm']:
        groups.append((f'loss_{family}', 'regression', {
            'smooth_l1': [(BASE,f'{family}_seed_{s}') for s in range(5)],
            **{criterion: [(CONTROL,f'loss_{family}_{criterion}_seed_{s}') for s in range(5)] for criterion in ['mae','mse']}}))
    for task in ['clf','reg']:
        groups.append((f'transfer_{task}', 'classification' if task=='clf' else 'regression',
            {variant: [(CONTROL,f'transfer_{task}_{variant}_seed_{s}') for s in range(5)] for variant in ['full','partial','scratch']}))
    with np.load(BASE/'runs/references/regression/training_mean.npz') as file:
        nonoverlap = file['nonoverlap_indices'].copy()
        regression_target = file['target'].reshape(-1).copy()
    support = pd.read_csv(CONTROL/'reports/large_target_support.csv')
    q90 = float(support.query('criterion=="training_q90"').iloc[0].threshold)
    seed_rows, intervals = [], []
    for group,task,variants in groups:
        while True:
            required=[parent/'runs'/name/'job_status.json' for locations in variants.values() for parent,name in locations]
            states=[json.loads(path.read_text())['state'] if path.exists() else 'pending' for path in required]
            if 'failed' in states:
                raise RuntimeError(f'A run in {group} failed; do not silently omit it')
            if all(state=='completed' for state in states):
                break
            if not args.wait:
                raise RuntimeError(f'All five seeds in every variant of {group} must complete')
            time.sleep(30)
        predictions = {};target = None
        for variant,locations in variants.items():
            values=[]
            for seed,(parent,name) in enumerate(locations):
                y,prediction = load(name,parent,task)
                if target is None:
                    target = y
                np.testing.assert_allclose(y,target,rtol=0,atol=1e-5)
                values.append(prediction)
                scopes = {'all':np.arange(len(y))}
                if task=='regression':
                    np.testing.assert_allclose(y,regression_target,rtol=0,atol=1e-5)
                    scopes.update(nonoverlap=nonoverlap, target_ge_6=np.flatnonzero(np.round(y,3)>=6),
                                  training_q90=np.flatnonzero(np.round(y,3)>=q90))
                for scope,index in scopes.items():
                    row=dict(group=group,task=task,variant=variant,seed=seed,scope=scope,n=len(index))
                    if task=='regression':
                        error=prediction[index]-y[index]
                        row.update(RMSE=float(np.sqrt(np.mean(error**2))) if len(index) else None,
                                   MAE=float(np.mean(abs(error))) if len(index) else None,
                                   fraction_error_above_1p8=float(np.mean(abs(error)>1.8)) if len(index) else None)
                    else:
                        row.update(AP=average_precision_score(y,prediction),AUC=roc_auc_score(y,prediction))
                    seed_rows.append(row)
            predictions[variant]=np.asarray(values)
        write_scores(seed_rows)
        for metric in (['rmse','mae'] if task=='regression' else ['ap','auc']):
            for block in [4,9,18,36]:
                result=paired_metric_bootstrap(y_true=target,metric_name=metric,task=task,
                    metric_config=BootstrapConfig(n_resamples=2000,seed=20261102,sampling='block',block_size=block),
                    baseline_model=next(iter(predictions)),
                    use_hierarchical_seed_sample=True,seed_aggregation='mean',point_estimate_mode='mean',
                    model_seed_predictions=predictions)
                for pair,delta in result.pairwise_deltas.items():
                    intervals.append(dict(group=group,task=task,metric=metric,left=pair[0],right=pair[1],
                        block_windows=block,delta=delta.point_estimate,low95=delta.ci_low,high95=delta.ci_high,
                        valid_resamples=delta.valid_resamples,total_resamples=delta.total_resamples,
                        sign='left minus right; lower regression error is better, higher AP/AUC is better'))
                pd.DataFrame(intervals).to_csv(OUT/'paired_block_intervals.csv',index=False)
        print(group,'completed',flush=True)
    summary=write_scores(seed_rows)
    (OUT/'status.json').write_text(json.dumps(dict(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
        control_jobs=60,seeds=[0,1,2,3,4],test_training_q90_windows=0,
        note='Source checkpoint held fixed. No seed filtering. Zero-support tail scores retained as missing.'),indent=2))
    print(summary.query('scope=="all"').to_string(index=False),flush=True)


if __name__=='__main__':
    main()
