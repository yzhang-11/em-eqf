"""Score frozen GAS choices, calibration, temporal variation and robustness."""
from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/run'));sys.path.insert(0,str(ROOT/'scripts/analyze'))
from run_magnitude_dynamics import OUT,REPORT,data_for
from src.models.tpp.magnitude_dynamics import filter_gas,LN10
from compare_magnitude_models import block_intervals,gr_logp


def ks_uniform(values):
    values=np.sort(values);n=len(values)
    return float(max(np.max(np.arange(1,n+1)/n-values),np.max(values-np.arange(n)/n)))


def gr_probabilities(b,data):
    rate=np.atleast_1d(b)*LN10
    edges=np.array([0.,.2,.5,1.,2.,data['width']])
    cdf=-np.expm1(-rate[:,None]*edges[None,:])/(-np.expm1(-rate[:,None]*data['width']))
    return np.diff(cdf,axis=1)


def b_variation(b,days):
    values=np.asarray(b,dtype=float)
    change=np.abs(np.diff(values))
    return dict(b_mean=float(values.mean()),b_sd=float(values.std()),
        mean_abs_delta_b=float(change.mean()),p95_abs_delta_b=float(np.quantile(change,.95)),
        b_total_variation_per_day=float(change.sum()/days))


def calibration(logp,probability,lower,data,mask):
    categories=np.searchsorted(np.rint(np.array([.2,.5,1.,2.,data['width']])/data['step']).astype(int),data['bins'],side='right')
    target=np.eye(5)[categories[mask]]
    prob=probability[mask]
    np.testing.assert_allclose(prob.sum(1),np.ones(mask.sum()),atol=1e-8)
    uniform=np.random.default_rng(20261104).random(mask.sum())
    pit=lower[mask]+uniform*np.exp(logp[mask])
    assert np.isfinite(pit).all() and np.min(pit)>=-1e-8 and np.max(pit)<=1.+1e-8
    tv=.5*np.abs(np.diff(prob,axis=0)).sum(1)
    metadata=data['periods']['test']
    return dict(nll_event=float(-logp[mask].mean()),coarse_brier=float(np.mean(np.sum((prob-target)**2,axis=1))),
        randomized_pit_KS=ks_uniform(pit),mean_probability_TV=float(tv.mean()),p95_probability_TV=float(np.quantile(tv,.95)),
        probability_TV_per_day=float(tv.sum()/metadata['days']),events=int(mask.sum()))


def evaluate_family(row,data):
    family=row['family']
    trace=filter_gas(row['theta'],family,data['excess'],data['step'],data['width'],
        components=data['component_logp'],return_trace=True)
    if family=='mixture_contractive':
        weight=trace['value']
        prob=weight[:,None]*data['component_coarse_probability'][0]+(1.-weight[:,None])*data['component_coarse_probability'][1]
        cumulative=np.vstack([np.zeros(2),np.cumsum(data['component_grid_probability'],axis=0)[:-1]])[data['bins']]
        lower=weight*cumulative[:,0]+(1.-weight)*cumulative[:,1]
    else:
        rate=trace['value']*LN10
        prob=gr_probabilities(trace['value'],data)
        lower=-np.expm1(-rate*data['excess'])/(-np.expm1(-rate*data['width']))
    return trace,prob,lower


def main():
    selection=json.loads((REPORT/'selection.json').read_text())
    assert selection['state']=='frozen_before_current_candidate_test_scoring'
    rows=[];intervals=[];b_rows=[]
    folder=REPORT/'predictions';folder.mkdir(exist_ok=True)
    for region in ['swc','scsn']:
        for variant in ['main','sensitivity']:
            data=data_for(region,variant)
            data['periods']=json.loads((OUT/'features'/f'{region}_{variant}_events.json').read_text())['periods']
            mask=data['split']==3;train=data['split']==1
            original=[];previous=[]
            for seed in range(5):
                with np.load(OUT/'features'/f'{region}_{variant}_seed_{seed}.npz') as d:
                    lp=d['old_logp'].copy();b=d['old_b'].copy()
                original.append(lp[mask])
                rate=b*LN10
                lower=-np.expm1(-rate*data['excess'])/(-np.expm1(-rate*data['width']))
                metrics=calibration(lp,gr_probabilities(b,data),lower,data,mask)
                rows.append(dict(region=region,variant=variant,model='original_GR',fit=seed,**metrics))
                b_rows.append(dict(region=region,variant=variant,model='original_GR',fit=seed,
                    **b_variation(b[mask],data['periods']['test']['days'])))
                with np.load(OUT/'reports/comparison'/f'{region}_{variant}_seed_{seed}.npz') as d:
                    previous.append(d['selected_logp'].copy())
            for row in selection['best']:
                if (row['region'],row['variant'])!=(region,variant):continue
                trace,probability,lower=evaluate_family(row,data)
                family=row['family']
                metrics=calibration(trace['logp'],probability,lower,data,mask)
                rows.append(dict(region=region,variant=variant,model=family,fit=0,selected=selection['choices'][region]==family,**metrics))
                if family!='mixture_contractive':
                    label='GR_GAS' if selection['choices'][region]==family else family
                    b_rows.append(dict(region=region,variant=variant,model=label,fit=0,
                        **b_variation(trace['value'][mask],data['periods']['test']['days'])))
                np.savez_compressed(folder/f'{region}_{variant}_{family}.npz',
                    times=data['times'][mask],magnitude=data['magnitude'][mask],logp=trace['logp'][mask],
                    state=trace['state'][mask],value=trace['value'][mask],coarse_probability=probability[mask])
                for reference,values in [('original_GR',np.asarray(original)),('previous_selected',np.asarray(previous))]:
                    gain=trace['logp'][mask][None,:]-values
                    for item in block_intervals(gain,data['times'][mask],data['periods']['test']['start'],data['periods']['test']['end']):
                        intervals.append(dict(region=region,variant=variant,model=family,reference=reference,**item))
            result=minimize_scalar(lambda rate:-gr_logp(rate,data['excess'][train],data['step'],data['width']).mean(),
                bounds=(LN10*.2,LN10*2.),method='bounded')
            rate=result.x
            lp=gr_logp(rate,data['excess'],data['step'],data['width'])
            lower=-np.expm1(-rate*data['excess'])/(-np.expm1(-rate*data['width']))
            prob=np.broadcast_to(gr_probabilities(rate/LN10,data),(len(lp),5))
            rows.append(dict(region=region,variant=variant,model='static_GR',fit=0,**calibration(lp,prob,lower,data,mask)))
            weight=1./(1.+np.exp(-float(data['static_logit'])))
            lp=np.logaddexp(np.log(weight)+data['component_logp'][:,0],np.log1p(-weight)+data['component_logp'][:,1])
            prob=np.broadcast_to(weight*data['component_coarse_probability'][0]+(1-weight)*data['component_coarse_probability'][1],(len(lp),5))
            cumulative=np.vstack([np.zeros(2),np.cumsum(data['component_grid_probability'],axis=0)[:-1]])[data['bins']]
            lower=weight*cumulative[:,0]+(1-weight)*cumulative[:,1]
            rows.append(dict(region=region,variant=variant,model='static_Kumaraswamy',fit=0,**calibration(lp,prob,lower,data,mask)))
    frame=pd.DataFrame(rows)
    frame.to_csv(REPORT/'test_scores.csv',index=False)
    summary=frame.groupby(['region','variant','model']).agg(
        **{key:(key,'mean') for key in ['nll_event','coarse_brier','randomized_pit_KS','mean_probability_TV','p95_probability_TV','probability_TV_per_day','events']},
        fits=('fit','size')).reset_index()
    summary.to_csv(REPORT/'test_summary.csv',index=False)
    pd.DataFrame(b_rows).to_csv(REPORT/'b_variation_fit_scores.csv',index=False)
    pd.DataFrame(b_rows).groupby(['region','variant','model']).agg(
        **{key:(key,'mean') for key in ['b_mean','b_sd','mean_abs_delta_b','p95_abs_delta_b','b_total_variation_per_day']},
        fits=('fit','size')).reset_index().to_csv(REPORT/'b_variation_summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(REPORT/'paired_block_intervals.csv',index=False)
    initialization=pd.read_csv(REPORT/'initialization_validation_scores.csv')
    stability=initialization.groupby(['region','variant','family']).agg(
        starts=('initial_index','size'),converged=('optimizer_success','sum'),
        minimum_validation=('validation_nll','min'),maximum_validation=('validation_nll','max'),
        minimum_train=('train_nll','min'),maximum_train=('train_nll','max')).reset_index()
    stability['validation_range']=stability.maximum_validation-stability.minimum_validation
    stability.to_csv(REPORT/'initialization_stability_summary.csv',index=False)
    print(summary.to_string(index=False))
    print(stability.to_string(index=False))
    print(pd.DataFrame(intervals).query('block_segments==4 and reference=="original_GR"').to_string(index=False))


if __name__=='__main__':main()
