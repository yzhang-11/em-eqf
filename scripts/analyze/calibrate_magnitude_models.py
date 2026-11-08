"""Matched randomized PIT, coarse probability Brier and temporal variation diagnostics."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/run'))
from src.models.tpp.magnitude_density import MagnitudeDensityHead
from run_magnitude_models import OUT,load_data


def ks_uniform(values):
    x=np.sort(values);n=len(x)
    return float(max(np.max(np.arange(1,n+1)/n-x),np.max(x-np.arange(n)/n)))


def main():
    torch.set_num_threads(1)
    selection=json.loads((OUT/'selection.json').read_text());rows=[];histograms=[];tails=[]
    for region in ['swc','scsn']:
        recipe=selection['recipes'][region]['selected'];family,k,mode=recipe['family'],recipe['components'],recipe['features']
        for variant in ['main','sensitivity']:
            for seed in range(5):
                run=OUT/'runs'/f'confirm_{region}_{variant}_{family}{k}_{mode}_seed_{seed}'
                saved=torch.load(run/'best_head.pth',map_location='cpu',weights_only=False)
                data=load_data(region,variant,seed,mode);mask=data['split']==3
                mc,step=float(data['Mc']),float(data['step']);width=10.-mc
                x=torch.tensor(((data['features']-saved['normalization_mean'])/saved['normalization_std'])[mask])
                y=torch.tensor(mc+data['bins'][mask]*step,dtype=torch.float64)
                head=MagnitudeDensityHead(saved['input_dim'],family,k,mc,step,hidden=32);head.load_state_dict(saved['state_dict']);head.eval()
                with torch.no_grad():
                    dist=head(x)
                    lp=dist.log_prob(y).numpy()
                    cdf_lower=dist.cdf(y-step).numpy()
                    edges=np.array([mc,mc+.2,mc+.5,mc+1.,mc+2.,10.])
                    upper=torch.tensor(edges[1:]-step).unsqueeze(-1)
                    candidate_cdf=dist.cdf(upper).numpy().T
                rate=data['old_b'][mask]*np.log(10.) if 'old_b' in data else None
                with np.load(OUT/'features'/f'{region}_{variant}_seed_{seed}.npz') as d:rate=d['old_b'][mask]*np.log(10.)
                z=-np.expm1(-rate*width)
                for threshold in [mc+.5,mc+1.,5.,6.]:
                    if threshold<=mc:continue
                    with torch.no_grad():
                        candidate_tail=1.-dist.cdf(torch.full_like(y,threshold-step)).numpy()
                    original_tail=(np.exp(-rate*(threshold-mc))-np.exp(-rate*width))/z
                    truth=y.numpy()>=threshold-1e-8
                    for label,prob in [('original',original_tail),('selected',candidate_tail)]:
                        assert np.all((prob>=-1e-8)&(prob<=1.+1e-8))
                        tails.append(dict(region=region,variant=variant,seed=seed,model=label,
                            threshold=threshold,events=len(y),exceedances=int(truth.sum()),
                            predicted_exceedances=float(prob.sum()),brier=float(np.mean((prob-truth)**2))))
                original_cdf=(-np.expm1(-rate[:,None]*(edges[1:]-mc)[None,:]))/z[:,None]
                original_lower=-np.expm1(-rate*data['bins'][mask]*step)/z
                uniform=np.random.default_rng(20261104+seed).random(len(y))
                pit={'selected':cdf_lower+uniform*np.exp(lp),'original':original_lower+uniform*np.exp(data['old_logp'][mask])}
                probabilities={'selected':np.diff(np.c_[np.zeros(len(y)),candidate_cdf],axis=1),
                               'original':np.diff(np.c_[np.zeros(len(y)),original_cdf],axis=1)}
                category=np.searchsorted(edges[1:],y.numpy(),side='right');target=np.eye(len(edges)-1)[category]
                for label,prob in probabilities.items():
                    assert np.min(prob)>-1e-9 and np.max(abs(prob.sum(1)-1.))<1e-8
                    tv=.5*np.abs(np.diff(prob,axis=0)).sum(1)
                    rows.append(dict(region=region,variant=variant,seed=seed,model=label,events=len(y),
                        coarse_brier=float(np.mean(np.sum((prob-target)**2,axis=1))),
                        randomized_pit_KS=ks_uniform(pit[label]),mean_probability_TV=float(tv.mean()),
                        median_probability_TV=float(np.median(tv)),p95_probability_TV=float(np.quantile(tv,.95))))
                    hist,_=np.histogram(pit[label],bins=np.linspace(0,1,11))
                    for j,count in enumerate(hist):histograms.append(dict(region=region,variant=variant,seed=seed,model=label,bin=j,count=int(count)))
    out=OUT/'reports/comparison'
    pd.DataFrame(rows).to_csv(out/'calibration_seed_scores.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['region','variant','model']).mean(numeric_only=True).reset_index()
    summary.to_csv(out/'calibration_summary.csv',index=False)
    pd.DataFrame(histograms).to_csv(out/'matched_pit_histograms.csv',index=False)
    pd.DataFrame(tails).to_csv(out/'tail_brier_matched_seed_scores.csv',index=False)
    pd.DataFrame(tails).groupby(['region','variant','model','threshold']).agg(
        brier=('brier','mean'),predicted_exceedances=('predicted_exceedances','mean'),
        events=('events','first'),exceedances=('exceedances','first')).reset_index().to_csv(
        out/'tail_brier_matched_summary.csv',index=False)
    print(summary[['region','variant','model','coarse_brier','randomized_pit_KS','mean_probability_TV','p95_probability_TV']].to_string(index=False))


if __name__=='__main__':main()
