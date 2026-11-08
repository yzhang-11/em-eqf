"""Validation-only search of globally shared shapes with neural mixture gates."""
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

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/run'))
from src.models.tpp.magnitude_gated import GatedMagnitudeHead
from run_magnitude_models import OUT,load_data,save,digest


def fit(region,variant,seed,recipe,stage):
    name=f"{stage}_{region}_{variant}_{recipe['family']}2_{recipe['features']}_seed_{seed}"
    run=OUT/'runs'/name;run.mkdir(exist_ok=True)
    status_path=run/'job_status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text())
        assert status['state']=='completed',name
        return status
    config=dict(region=region,variant=variant,seed=seed,**recipe,components=2,hidden=32,
        maximum_epochs=400,patience=35,learning_rate=.003,weight_decay=0.,dropout=0.,
        loss='negative log reported-bin probability only',b_weight=0.,b_smooth_weight=0.,
        encoder='frozen; cached causal contexts',stage=stage,model_type='gated_magnitude',
        shapes='learned globally; only mixture weights vary across events',test_used_for_selection=False)
    save(OUT/'configs'/f'{name}.json',config)
    save(status_path,dict(state='running',name=name))
    started=time.monotonic();torch.manual_seed(20261104+seed);np.random.seed(20261104+seed)
    data=load_data(region,variant,seed,recipe['features'])
    train,val=data['split']==1,data['split']==2
    mean=data['features'][train].mean(0);std=np.maximum(data['features'][train].std(0),1e-5)
    x=(data['features']-mean)/std
    xtr=torch.tensor(x[train],device='cuda');xva=torch.tensor(x[val],device='cuda')
    ytr=torch.tensor(data['magnitude'][train],device='cuda');yva=torch.tensor(data['magnitude'][val],device='cuda')
    model=GatedMagnitudeHead(x.shape[1],recipe['family'],2,float(data['Mc']),float(data['step'])).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=0.)
    best=float('inf');best_state=None;stale=0;curve=[];best_epoch=0
    for epoch in range(401):
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            loss=-model(xtr).log_prob(ytr).mean();assert torch.isfinite(loss),name
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        if epoch%2:continue
        with torch.no_grad():
            validation=float(-model(xva).log_prob(yva).mean())
            training=float(-model(xtr).log_prob(ytr).mean())
        curve.append(dict(epoch=epoch,train=training,validation=validation))
        if validation<best-1e-6:
            best,best_epoch,stale=validation,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=2
        if stale>=35:break
    checkpoint=run/'best_head.pth'
    torch.save(dict(state_dict=best_state,normalization_mean=mean,normalization_std=std,config=config,
        Mc=float(data['Mc']),step=float(data['step']),best_epoch=best_epoch,validation_nll=best,input_dim=x.shape[1],
        model_type='gated_magnitude',base_checkpoint=str(ROOT/'experiments/revision/main/runs'/
        f'generation_{region}_{variant}_seed_{seed}'/'best_model_1.pth')),checkpoint)
    pd.DataFrame(curve).to_csv(run/'learning_curve.csv',index=False)
    status=dict(state='completed',name=name,region=region,variant=variant,seed=seed,**recipe,
        validation_nll=best,original_validation_nll=float(-data['old_logp'][val].mean()),
        validation_gain=float(-data['old_logp'][val].mean())-best,best_epoch=best_epoch,
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),elapsed_seconds=time.monotonic()-started,
        test_used_for_selection=False,model_type='gated_magnitude')
    save(status_path,status);print(json.dumps(status),flush=True)
    del model,optimizer;torch.cuda.empty_cache()
    return status


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm']);args=parser.parse_args()
    torch.set_num_threads(1)
    if args.stage=='screen':
        plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),
            model='shared global component shapes; neural causal mixture gates; NLL only',
            motivation='exploratory response to increased distribution variation and inconclusive generation checks',
            already_seen='original test and SCSN 2020-2024 results already inspected; neither used for gated fitting/selection',
            selection='training/validation only; no gated test scoring until confirmation and selection are frozen',
            families=['weibull','kumaraswamy'],features=['context','history','context_history'],
            source_files={p:digest(ROOT/p) for p in ['src/models/tpp/magnitude_gated.py','scripts/run/run_magnitude_gated.py']})
        save(OUT/'configs/gated_plan.json',plan)
        rows=[];recipes={}
        for region in ['swc','scsn']:
            for family in plan['families']:
                for features in plan['features']:
                    rows.append(fit(region,'main',0,dict(family=family,features=features),'gated_screen'))
                    pd.DataFrame(rows).to_csv(OUT/'reports/gated_validation_scores.csv',index=False)
            best=min([x for x in rows if x['region']==region],key=lambda x:x['validation_nll'])
            recipes[region]={k:best[k] for k in ['family','features']}
        save(OUT/'configs/gated_confirmation_recipes.json',recipes)
    else:
        recipes=json.loads((OUT/'configs/gated_confirmation_recipes.json').read_text());rows=[]
        for region in ['swc','scsn']:
            for variant in ['main','sensitivity']:
                for seed in range(5):
                    rows.append(fit(region,variant,seed,recipes[region],'gated_confirm'))
                    pd.DataFrame(rows).to_csv(OUT/'reports/gated_confirmation_validation_scores.csv',index=False)
        save(OUT/'configs/gated_selection.json',dict(state='frozen_before_gated_test_scoring',
            created_utc=datetime.now(timezone.utc).isoformat(),recipes=recipes,
            scope='exploratory architecture; previous candidate test and external periods already seen'))


if __name__=='__main__':main()
