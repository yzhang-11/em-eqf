"""Validation-only search of likelihood-trained magnitude-state models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts/run'))
from src.models.tpp.magnitude_state import MagnitudeStateModel
from run_magnitude_models import OUT, save, digest


def fit(region,variant,states,persistence,seed,stage):
    name=f'{stage}_{region}_{variant}_state{states}_p{persistence}_seed_{seed}'
    run=OUT/'runs'/name;run.mkdir(parents=True,exist_ok=True)
    status_path=run/'job_status.json'
    if status_path.exists():
        old=json.loads(status_path.read_text())
        assert old['state']=='completed',name
        return old
    with np.load(OUT/'features'/f'{region}_{variant}_events.npz') as d:data={k:d[k].copy() for k in d.files}
    with np.load(OUT/'features'/f'{region}_{variant}_seed_0.npz') as d:old_logp=d['old_logp'].copy()
    train=data['split']==1;val=data['split']==2
    train_end=np.flatnonzero(train)[-1]+1;val_end=np.flatnonzero(val)[-1]+1
    config=dict(region=region,variant=variant,states=states,persistence=persistence,seed=seed,
                learning_rate=.015,maximum_epochs=240,patience=30,loss='conditional reported-mark NLL only',
                b_weight=0.,b_smooth_weight=0.,weight_decay=0.,test_used_for_selection=False)
    save(OUT/'configs'/f'{name}.json',config)
    save(status_path,dict(state='running',config=config))
    device=torch.device('cuda:0');torch.manual_seed(20261104+seed)
    model=MagnitudeStateModel(states,float(data['Mc']),float(data['step']),persistence).to(device)
    with torch.no_grad():model.emission_parameters[:,1:].add_(torch.randn_like(model.emission_parameters[:,1:])*.03)
    ytr=torch.tensor(data['magnitude'][:train_end],device=device)
    yva=torch.tensor(data['magnitude'][:val_end],device=device)
    trmask=torch.tensor(train[:train_end],device=device)
    vamask=torch.tensor(val[:val_end],device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=0.)
    best=float('inf');best_epoch=0;best_state=None;stale=0;curve=[];start=time.monotonic()
    for epoch in range(config['maximum_epochs']+1):
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            loss=-model.log_prob_sequence(ytr)[trmask].mean()
            assert torch.isfinite(loss),name
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        if epoch%3:continue
        with torch.no_grad():
            validation=float(-model.log_prob_sequence(yva)[vamask].mean())
        curve.append(dict(epoch=epoch,validation_nll=validation))
        if validation<best-1e-6:
            best,best_epoch,stale=validation,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=3
        if stale>=config['patience']:break
    model.load_state_dict(best_state)
    checkpoint=run/'best_head.pth'
    torch.save(dict(state_dict=best_state,config=config,Mc=float(data['Mc']),step=float(data['step']),best_epoch=best_epoch,
                    model_type='magnitude_state',validation_nll=best),checkpoint)
    pd.DataFrame(curve).to_csv(run/'learning_curve.csv',index=False)
    status=dict(state='completed',name=name,region=region,variant=variant,states=states,persistence=persistence,seed=seed,
                validation_nll=best,original_validation_nll=float(-old_logp[val].mean()),
                validation_gain=float(-old_logp[val].mean())-best,best_epoch=best_epoch,
                transition=model.transition_logits.softmax(-1).detach().cpu().tolist(),
                checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),elapsed_seconds=time.monotonic()-start,
                test_used_for_selection=False)
    save(status_path,status);print(json.dumps(status),flush=True)
    del model,optimizer,ytr,yva,trmask,vamask;torch.cuda.empty_cache()
    return status


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm']);args=parser.parse_args()
    torch.set_num_threads(1)
    if args.stage=='screen':
        plan=dict(model='Markov-switching bounded magnitude densities; no neural b prediction or loss regularizers',
            states=[2,3],initial_persistence=[.95,.995],selection='validation likelihood; test never evaluated',
            source_files={p:digest(ROOT/p) for p in ['src/models/tpp/magnitude_state.py','scripts/run/run_magnitude_states.py']})
        save(OUT/'configs/state_plan.json',plan)
        results=[];selected={}
        for region in ['swc','scsn']:
            candidates=[]
            for states in [2,3]:
                for persistence in [.95,.995]:
                    main=fit(region,'main',states,persistence,0,'state_screen');results.append(main)
                    sensitivity=fit(region,'sensitivity',states,persistence,0,'state_screen');results.append(sensitivity)
                    candidates.append(dict(states=states,persistence=persistence,
                        mean_validation_gain=.5*(main['validation_gain']+sensitivity['validation_gain'])))
                    pd.DataFrame(results).to_csv(OUT/'reports/state_validation_scores.csv',index=False)
            selected[region]=max(candidates,key=lambda x:x['mean_validation_gain'])
        save(OUT/'configs/state_confirmation_recipes.json',selected)
    else:
        selected=json.loads((OUT/'configs/state_confirmation_recipes.json').read_text());results=[]
        for region in ['swc','scsn']:
            for variant in ['main','sensitivity']:
                for seed in range(5):
                    recipe=selected[region]
                    results.append(fit(region,variant,recipe['states'],recipe['persistence'],seed,'state_confirm'))
                    pd.DataFrame(results).to_csv(OUT/'reports/state_confirmation_scores.csv',index=False)
        summary=pd.DataFrame(results).groupby(['region','variant']).agg(validation=('validation_nll','mean'),gain=('validation_gain','mean'))
        summary.to_csv(OUT/'reports/state_confirmation_summary.csv');print(summary.to_string(),flush=True)


if __name__=='__main__':main()
