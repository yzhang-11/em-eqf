"""Audit frozen thresholds and summarize paired, physically scaled predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
BASE = ROOT / 'experiments/revision/main'
OUT = BASE / 'reports/supervised'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary=json.loads((BASE/'runner_summary.json').read_text())
    assert summary['completed']==65 and summary['failed']==0
    regressions, prediction={},{}
    target=None
    for name in ['reg_azdx','reg_scratch','lstm']:
        values=[]
        for seed in range(5):
            folder=BASE/'runs'/f'{name}_seed_{seed}'
            with np.load(folder/'predictions_test.npz') as file:
                y,p=file['target'].reshape(-1),file['prediction'].reshape(-1)
            if target is None:target=y
            np.testing.assert_allclose(y,target,rtol=0,atol=1e-5)
            values.append(p)
        prediction[name]=np.asarray(values)
    baseline=BASE/'runs/references/regression'
    for name in ['training_mean','training_median','past_180_day_maximum']:
        with np.load(baseline/f'{name}.npz') as file:
            np.testing.assert_allclose(file['target'],target,rtol=0,atol=1e-5)
            prediction[name]=file['prediction'].reshape(1,-1)
            origins=file['origin_day'].copy()
            nonoverlap=file['nonoverlap_indices'].copy()
    rows=[]
    for name,p in prediction.items():
        for scope,index in [('all',np.arange(len(target))),('nonoverlap',nonoverlap),
                            ('target_ge_6',np.flatnonzero(np.round(target,3)>=6.))]:
            error=p[:,index]-target[index]
            rmse=np.sqrt(np.mean(error**2,axis=1));mae=np.mean(abs(error),axis=1)
            rows.append(dict(model=name,scope=scope,n=len(index),seeds=len(p),
                             RMSE_mean=float(rmse.mean()),RMSE_sd=float(rmse.std(ddof=1)) if len(p)>1 else None,
                             MAE_mean=float(mae.mean()),fraction_error_above_1p8=float(np.mean(abs(error)>1.8))))
    pd.DataFrame(rows).to_csv(OUT/'regression_reference_and_tail_scores.csv',index=False)

    # Difference of seed-wise RMSE means, sharing time indices across all models.
    intervals=[];n=len(target);replicates=2000
    em_error=(prediction['reg_azdx']-target)**2
    for name,p in prediction.items():
        if name=='reg_azdx':continue
        reference_error=(p-target)**2
        for block in [4,9,18,36]:
            rng=np.random.default_rng(20261102)
            starts=rng.integers(0,n,size=(replicates,int(np.ceil(n/block))))
            time_index=((starts[:,:,None]+np.arange(block))%n).reshape(replicates,-1)[:,:n]
            seeds=rng.integers(0,5,size=(replicates,5))
            other_seeds=seeds if name=='reg_scratch' else rng.integers(0,len(p),size=(replicates,len(p)))
            em_boot=np.sqrt(em_error[seeds[:,:,None],time_index[:,None,:]].mean(axis=2)).mean(axis=1)
            ref_boot=np.sqrt(reference_error[other_seeds[:,:,None],time_index[:,None,:]].mean(axis=2)).mean(axis=1)
            delta=ref_boot-em_boot
            point=np.sqrt(reference_error.mean(axis=1)).mean()-np.sqrt(em_error.mean(axis=1)).mean()
            low,high=np.quantile(delta,[.025,.975])
            intervals.append(dict(reference=name,block_windows=block,block_shift_days=block*10,replicates=replicates,
                                  effective_blocks=n/block,delta_RMSE=float(point),low95=float(low),high95=float(high),
                                  sign='positive favors pretrained model; paired time blocks; source fit held fixed'))
    pd.DataFrame(intervals).to_csv(OUT/'regression_paired_block_intervals.csv',index=False)

    classification=[];thresholds=[]
    for horizon,mf in [(10,4.),(20,4.5),(30,4.5),(60,5.),(90,5.5)]:
        common_y=None
        for model in ['azdx','scratch','rf']:
            for seed in range(5):
                if model=='rf':
                    folder=BASE/'runs/references'/f'rf_h{horizon}_seed_{seed}'
                    metrics=json.loads((folder/'metrics_test.json').read_text())
                    threshold=json.loads((folder/'metrics_validation.json').read_text())['threshold']
                else:
                    folder=BASE/'runs'/f'clf_{model}_h{horizon}_seed_{seed}'
                    metrics=json.loads((folder/'metrics_test_best_1.json').read_text())
                    checkpoint=torch.load(folder/'best_model_1.pth',map_location='cpu',weights_only=False)
                    threshold=checkpoint['val_metrics']['threshold']
                with np.load(folder/'predictions_test.npz') as file:
                    y=file['target'].reshape(-1)
                    score=expit(file['logits'].reshape(-1)) if 'logits' in file else file['score'].reshape(-1)
                    assert abs(float(file['threshold'])-threshold)<1e-12
                if common_y is None:common_y=y
                np.testing.assert_array_equal(y,common_y)
                values=dict(auc=roc_auc_score(y,score),pr_auc=average_precision_score(y,score),
                            f1=f1_score(y,score>threshold,zero_division=0),R=metrics['R'])
                for key in ['auc','pr_auc','f1']:
                    assert abs(values[key]-metrics[key])<1e-8,(folder,key,values[key],metrics[key])
                classification.append(dict(horizon=horizon,Mf=mf,model=model,seed=seed,n=len(y),positive=int(y.sum()),
                                           prevalence=float(y.mean()),**values))
                thresholds.append(dict(horizon=horizon,model=model,seed=seed,threshold=threshold,
                                       threshold_source='validation',test_threshold_matches=True))
    frame=pd.DataFrame(classification)
    frame.to_csv(OUT/'classification_seed_scores.csv',index=False)
    frame.groupby(['horizon','Mf','model']).agg(n=('n','first'),positive=('positive','first'),prevalence=('prevalence','first'),
                    AP_mean=('pr_auc','mean'),AP_sd=('pr_auc','std'),AUC_mean=('auc','mean'),AUC_sd=('auc','std'),
                    F1_mean=('f1','mean'),R_mean=('R','mean')).reset_index().to_csv(OUT/'classification_summary.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'classification_validation_threshold_audit.csv',index=False)
    print(pd.DataFrame(rows).query('scope=="all"').to_string(index=False))
    print(pd.DataFrame(intervals).query('block_windows==18').to_string(index=False))
    print('Threshold and label audits passed:',len(thresholds),'model-seed settings')


if __name__=='__main__':
    main()
