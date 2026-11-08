"""Fit low-dimensional GAS states and audit initialization/scale sensitivity."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.models.tpp.magnitude_dynamics import filter_gas,parameters,contraction_bounds,LN10

OUT=ROOT/'experiments/revision/magnitude_models'
REPORT=OUT/'reports/dynamics'
FAMILIES=['gr_free','gr_contractive','mixture_contractive']
CAPS={'gr_free':3.,'gr_contractive':.3,'mixture_contractive':2.}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def data_for(region,variant):
    with np.load(OUT/'features'/f'{region}_{variant}_events.npz') as f:
        data={key:f[key].copy() for key in f.files}
    with np.load(REPORT/'inputs'/f'{region}_{variant}_components.npz') as f:
        data.update({key:f[key].copy() for key in f.files})
    data['step']=float(data['step']);data['Mc']=float(data['Mc'])
    data['excess']=data['bins']*data['step'];data['width']=10.-data['Mc']
    return data


def prepare():
    sys.path.insert(0,str(ROOT/'scripts/run'))
    from run_magnitude_models import train_fit
    from src.models.tpp.magnitude_density import MagnitudeDensityHead,MagnitudeGridDistribution
    import torch
    torch.set_num_threads(1)
    for region in ['swc','scsn']:
        for variant in ['main','sensitivity']:
            status=train_fit(region,variant,0,dict(family='kumaraswamy',components=2,features='static'),'dynamics_static')
            path=Path(status['checkpoint']);saved=torch.load(path,map_location='cpu',weights_only=False)
            head=MagnitudeDensityHead(0,'kumaraswamy',2,saved['Mc'],saved['step'])
            head.load_state_dict(saved['state_dict']);head.eval()
            with torch.no_grad():raw=head.network(torch.ones(1,1))[0]
            component_raw=torch.stack([torch.stack([raw.new_tensor(0.),raw[2+j],raw[4+j]]) for j in range(2)])
            dist=MagnitudeGridDistribution(component_raw,'kumaraswamy',1,saved['Mc'],saved['step'])
            with np.load(OUT/'features'/f'{region}_{variant}_events.npz') as f:
                bins=f['bins'].copy()
            grid=np.arange(round((10.-saved['Mc'])/saved['step']))
            with torch.no_grad():
                component_grid_logp=dist.log_prob(torch.tensor(saved['Mc']+grid*saved['step'])[:,None]).numpy()
            component_grid_probability=np.exp(component_grid_logp)
            np.testing.assert_allclose(component_grid_probability.sum(0),np.ones(2),atol=2e-10)
            cut=np.rint(np.array([.2,.5,1.,2.,10.-saved['Mc']])/saved['step']).astype(int)
            category=np.searchsorted(cut,grid,side='right')
            coarse=np.stack([np.bincount(category,weights=row,minlength=5) for row in component_grid_probability.T])
            folder=REPORT/'inputs';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f'{region}_{variant}_components.npz',
                component_logp=component_grid_logp[bins],component_grid_probability=component_grid_probability,
                component_coarse_probability=coarse,static_logit=float(raw[0]-raw[1]),
                static_checkpoint_sha256=digest(path))
            save(folder/f'{region}_{variant}_components.json',dict(checkpoint=str(path),sha256=digest(path),
                recipe='two global Kumaraswamy components trained with NLL; validation early stopping; fixed for GAS',
                raw=raw.tolist()))


def fit(region,variant,family,initial_index,cap=None,start=None):
    cap=CAPS[family] if cap is None else float(cap)
    suffix='' if cap==CAPS[family] else f'_cap_{cap:g}'
    name=f'dynamics_{region}_{variant}_{family}{suffix}_init_{initial_index}'
    run=OUT/'runs'/name;run.mkdir(exist_ok=True)
    status_path=run/'job_status.json'
    if status_path.exists():
        old=json.loads(status_path.read_text());assert old['state']=='completed',name
        return old
    data=data_for(region,variant)
    train=data['split']==1;val=data['split']==2
    end=np.flatnonzero(train)[-1]+1
    x=data['excess'][:end];mask=train[:end];comp=data['component_logp'][:end]
    static=minimize_scalar(lambda rate:float((rate*x[mask]-np.log(-np.expm1(-rate*data['step']))+
        np.log(-np.expm1(-rate*data['width']))).mean()),bounds=(LN10*.2,LN10*2.),method='bounded')
    b=static.x/LN10;fraction=np.clip((b-.2)/1.8,1e-6,1.-1e-6)
    center=float(data['static_logit']) if family=='mixture_contractive' else float(np.log(fraction/(1-fraction)))
    bounds=[(-8.,8.),(math.log(2.),math.log(100000.)),(math.log(1e-6),math.log(cap))]
    if start is None:
        life=[16.,64.,256.,1024.,4096.][initial_index]
        gain=[.002,.005,.01,.02,.05][initial_index] if family=='gr_free' else cap*[.05,.2,.5,.8,.95][initial_index]
        start=np.array([center,math.log(life),math.log(gain)])
    else:
        start=np.asarray(start).copy()
        start=np.array([min(hi,max(lo,value)) for value,(lo,hi) in zip(start,bounds)])
    config=dict(region=region,variant=variant,family=family,initial_index=initial_index,gain_cap=cap,
        start=start.tolist(),bounds=bounds,optimizer='L-BFGS-B',maximum_iterations=150,
        loss='proper reported-bin NLL only',b_weight=0.,b_smooth_weight=0.,
        selection='minimum training NLL across starts; validation selects family; test unused',
        memory='per-event; does not impose a calendar-time derivative bound',stage='dynamics')
    save(OUT/'configs'/f'{name}.json',config);save(status_path,dict(state='running',name=name))
    begin=time.monotonic()
    def objective(theta):
        return filter_gas(theta,family,x,data['step'],data['width'],mask,comp)
    result=minimize(objective,start,method='L-BFGS-B',bounds=bounds,
        options=dict(maxiter=150,maxls=30,ftol=1e-10,gtol=2e-6,eps=1e-5))
    assert np.isfinite(result.fun) and np.isfinite(result.x).all(),name
    val_end=np.flatnonzero(val)[-1]+1
    validation=filter_gas(result.x,family,data['excess'][:val_end],data['step'],data['width'],
        val[:val_end],data['component_logp'][:val_end])
    mean,rho,alpha,half_life,gain=parameters(result.x,family)
    lower,upper=contraction_bounds(result.x,family,data['width'])
    if family!='gr_free':assert max(abs(lower),abs(upper))<1.,(name,lower,upper)
    checkpoint=run/'parameters.json'
    save(checkpoint,dict(theta=result.x.tolist(),config=config,Mc=data['Mc'],step=data['step'],
        component_input=str(REPORT/'inputs'/f'{region}_{variant}_components.npz'),
        component_input_sha256=digest(REPORT/'inputs'/f'{region}_{variant}_components.npz')))
    status=dict(state='completed',name=name,region=region,variant=variant,family=family,
        initial_index=initial_index,gain_cap=cap,theta=result.x.tolist(),train_nll=float(result.fun),
        validation_nll=float(validation),optimizer_success=bool(result.success),optimizer_message=str(result.message),
        iterations=int(result.nit),evaluations=int(result.nfev),mean_state=mean,persistence=rho,
        innovation_coefficient=alpha,half_life_events=half_life,gain=gain,
        contraction_lower=lower,contraction_upper=upper,checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),elapsed_seconds=time.monotonic()-begin,
        test_used_for_selection=False,stage='dynamics')
    save(status_path,status)
    return status


def sensitivity(best):
    rows=[]
    for status in best:
        data=data_for(status['region'],status['variant']);mask=data['split']==2
        end=np.flatnonzero(mask)[-1]+1
        for axis in ['half_life','gain']:
            for factor in [.25,.5,1.,2.,4.]:
                theta=np.asarray(status['theta']).copy();index=1 if axis=='half_life' else 2
                theta[index]+=math.log(factor)
                _,_,_,life,gain=parameters(theta,status['family'])
                valid=2.<=life<=100000. and 1e-6<=gain<=status['gain_cap']*(1.+1e-12)
                value=filter_gas(theta,status['family'],data['excess'][:end],data['step'],data['width'],
                    mask[:end],data['component_logp'][:end]) if valid else np.nan
                rows.append(dict(region=status['region'],variant=status['variant'],family=status['family'],
                    axis=axis,multiplier=factor,within_model_constraints=valid,validation_nll=value))
    pd.DataFrame(rows).to_csv(REPORT/'scale_sensitivity_validation.csv',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();REPORT.mkdir(exist_ok=True)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        priority='accuracy plus stable distribution dynamics and broad hyperparameter sensitivity, not absence of every regularizer',
        models=FAMILIES,default_gain_caps=CAPS,half_life_bounds_events=[2.,100000.],
        initial_half_lives_events=[16.,64.,256.,1024.,4096.],
        alternative_caps={'gr_contractive':[.1,.2],'mixture_contractive':[.5,1.]},
        robustness='all five initializations; refit conservative gain caps; perturb learned scales by .25/.5/1/2/4 on validation',
        sensitivity_invalid='out-of-constraint settings recorded invalid, never clipped into duplicate settings',
        selection='best training likelihood start per family; family by main-Mc validation only; cap variants diagnostic only',
        heldout='previous test and 2020-2024 periods already inspected; all current candidates exploratory',
        metric='same .1 SWC/.01 SCSN reported bins and upper bound 10; probabilistic calibration and coarse probability TV',
        source_files={p:digest(ROOT/p) for p in ['src/models/tpp/magnitude_dynamics.py','scripts/run/run_magnitude_dynamics.py']})
    path=REPORT/'plan.json'
    if path.exists():
        previous=json.loads(path.read_text())
        for name,sha in previous['source_files'].items():assert digest(ROOT/name)==sha,name
    else:save(path,plan)
    prepare()
    jobs=[(r,v,f,s) for r in ['swc','scsn'] for v in ['main','sensitivity'] for f in FAMILIES for s in range(5)]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(fit,*job) for job in jobs]
        for future in as_completed(futures):
            status=future.result();results.append(status)
            pd.DataFrame(results).to_csv(REPORT/'initialization_validation_scores.csv',index=False)
            save(REPORT/'status.json',dict(state='fitting',completed=len(results),total=len(jobs)))
            print(json.dumps(status),flush=True)
    best=[]
    for region in ['swc','scsn']:
        for variant in ['main','sensitivity']:
            for family in FAMILIES:
                rows=[r for r in results if (r['region'],r['variant'],r['family'])==(region,variant,family)]
                best.append(min(rows,key=lambda r:r['train_nll']))
    pd.DataFrame(best).to_csv(REPORT/'best_training_start_validation_scores.csv',index=False)
    alternatives=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(fit,r['region'],r['variant'],r['family'],99,cap,r['theta'])
                 for r in best if r['family']!='gr_free' for cap in plan['alternative_caps'][r['family']]]
        for future in as_completed(futures):
            status=future.result();alternatives.append(status)
            pd.DataFrame(alternatives).to_csv(REPORT/'cap_sensitivity_validation_scores.csv',index=False)
            print(json.dumps(status),flush=True)
    sensitivity(best)
    choices={region:min([r for r in best if r['region']==region and r['variant']=='main'],
        key=lambda r:r['validation_nll'])['family'] for region in ['swc','scsn']}
    selection=dict(state='frozen_before_current_candidate_test_scoring',created_utc=datetime.now(timezone.utc).isoformat(),
        choices=choices,best=best,rule='family by main-Mc validation; fitting initialization by training likelihood',
        baseline_periods_previously_seen=True)
    save(REPORT/'selection.json',selection)
    save(REPORT/'status.json',dict(state='validation_completed',new_static_fits=4,
        default_optimizations=60,alternative_cap_optimizations=16,choices=choices))
    print(pd.DataFrame(best)[['region','variant','family','validation_nll','half_life_events','gain']].to_string(index=False),flush=True)


if __name__=='__main__':main()
