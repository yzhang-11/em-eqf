"""Score validation-selected magnitude heads with matched grid probabilities and paired calendar blocks."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/run'))
from src.models.tpp.magnitude_density import MagnitudeDensityHead
from run_magnitude_models import load_data, OUT


def gr_logp(rate, lower, step, width):
    return -rate*lower + np.log(-np.expm1(-rate*step)) - np.log(-np.expm1(-rate*width))


def references(data):
    mc, step = float(data['Mc']), float(data['step'])
    lower = data['bins']*step
    width = 10.-mc
    train, validation = data['split'] == 1, data['split'] == 2
    result = minimize_scalar(lambda rate:-gr_logp(rate,lower[train],step,width).mean(), bounds=(.02,15.), method='bounded')
    assert result.success
    rate = float(result.x)
    static = gr_logp(rate,lower,step,width)
    prior_n = 20.
    prior_k = 1./np.expm1(rate*step)
    choices = []
    for discount in [.95,.98,.995,.999]:
        n, total = prior_n, prior_n*prior_k
        rates = np.empty(len(lower))
        for i,k in enumerate(data['bins']):
            n *= discount;total *= discount
            q = np.clip(total/(total+n),1e-8,1.-1e-8)
            rates[i] = -np.log(q)/step
            n += 1.;total += float(k)
        values = gr_logp(rates,lower,step,width)
        choices.append((float(-values[validation].mean()),discount,values))
    best = min(choices,key=lambda x:x[0])
    return {'static_discrete_gr':static,'dynamic_discrete_gr':best[2]}, dict(static_rate=rate,selected_discount=best[1],
        discount_rule='minimum validation probability NLL; causal discounted counts before current event',
        candidates=[{'discount':x[1],'validation_nll':x[0]} for x in choices])


def block_intervals(gain, times, start, end):
    segment = np.minimum(((times-start)/(end-start)*20).astype(int),19)
    count = np.bincount(segment,minlength=20)
    sums = np.stack([np.bincount(segment,weights=row,minlength=20) for row in gain])
    rows=[]
    for block in [1,2,4]:
        rng=np.random.default_rng(20261104+block)
        origins=rng.integers(0,20,size=(2000,int(np.ceil(20/block))))
        index=((origins[:,:,None]+np.arange(block))%20).reshape(2000,-1)[:,:20]
        seeds=rng.integers(0,gain.shape[0],size=(2000,gain.shape[0]))
        values=sums[seeds[:,:,None],index[:,None,:]].sum(-1).mean(-1)/count[index].sum(-1)
        low,high=np.quantile(values,[.025,.975])
        rows.append(dict(block_segments=block,calendar_segments=20,resamples=2000,
                         gain_nats_event=float(gain.mean()),low95=float(low),high95=float(high)))
    return rows


def main():
    torch.set_num_threads(1)
    selected_path=OUT/'selection.json'
    selected=json.loads(selected_path.read_text())
    assert selected['state']=='frozen_before_candidate_test_scoring'
    report=OUT/'reports/comparison'
    report.mkdir(exist_ok=True)
    scores,intervals,pits,briers,reference_settings=[],[],[],[],[]
    device=torch.device('cuda:0')
    for region in ['swc','scsn']:
        recipe={k:selected['recipes'][region]['selected'][k] for k in ['family','components','features']}
        for variant in ['main','sensitivity']:
            candidate,original,control,features_data=[],[],[],[]
            for seed in range(5):
                name=f"confirm_{region}_{variant}_{recipe['family']}{recipe['components']}_{recipe['features']}_seed_{seed}"
                path=OUT/'runs'/name/'best_head.pth'
                saved=torch.load(path,map_location='cpu',weights_only=False)
                data=load_data(region,variant,seed,recipe['features'])
                mask=data['split']==3
                normalized=(data['features']-saved['normalization_mean'])/saved['normalization_std']
                model=MagnitudeDensityHead(saved['input_dim'],recipe['family'],recipe['components'],float(data['Mc']),float(data['step']),hidden=32).to(device)
                model.load_state_dict(saved['state_dict']);model.eval()
                x=torch.tensor(normalized[mask],device=device)
                y=torch.tensor(data['magnitude'][mask],device=device)
                with torch.inference_mode():
                    dist=model(x)
                    lp=dist.log_prob(y).cpu().numpy()
                    lo=dist.cdf(y.double()-float(data['step'])).cpu().numpy()
                    probability=dist.log_prob(y).exp().cpu().numpy()
                    uniform=np.random.default_rng(20261104+seed).random(len(y))
                    randomized=lo+uniform*probability
                    for threshold in [float(data['Mc'])+.5,float(data['Mc'])+1.,5.,6.]:
                        if threshold<=float(data['Mc']):continue
                        pred=1.-dist.cdf(torch.full_like(y,threshold-float(data['step']))).cpu().numpy()
                        truth=data['magnitude'][mask]>=threshold-1e-6
                        briers.append(dict(region=region,variant=variant,seed=seed,model='selected',threshold=threshold,brier=float(np.mean((pred-truth)**2)),events=int(truth.sum())))
                assert np.isfinite(lp).all() and np.all((randomized>=-1e-6)&(randomized<=1.+1e-6))
                candidate.append(lp);original.append(data['old_logp'][mask]);features_data.append(data)
                for model_name,values in [('selected',lp),('original',original[-1])]:
                    scores.append(dict(region=region,variant=variant,seed=seed,model=model_name,n=len(values),nll_event=float(-values.mean()),family=recipe['family'] if model_name=='selected' else 'original_GR'))
                pit_hist,_=np.histogram(randomized,bins=np.linspace(0,1,11))
                for k,value in enumerate(pit_hist):pits.append(dict(region=region,variant=variant,seed=seed,model='selected',bin=k,count=int(value),events=len(y)))
                np.savez_compressed(report/f'{region}_{variant}_seed_{seed}.npz',times=data['times'][mask],magnitude=data['magnitude'][mask],
                                    selected_logp=lp,original_logp=original[-1],pit=randomized,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                del model,dist,x,y
            reference,settings=references(features_data[0]);reference_settings.append(dict(region=region,variant=variant,**settings))
            mask=features_data[0]['split']==3
            for name,values in reference.items():scores.append(dict(region=region,variant=variant,seed=-1,model=name,n=int(mask.sum()),nll_event=float(-values[mask].mean()),family='discrete_GR_reference'))
            metadata=json.loads((OUT/'features'/f'{region}_{variant}_events.json').read_text())['periods']['test']
            references_for_intervals={'original':np.asarray(original)}
            references_for_intervals.update({name:np.broadcast_to(values[mask],np.asarray(candidate).shape) for name,values in reference.items()})
            for name,values in references_for_intervals.items():
                gain=np.asarray(candidate)-values
                for row in block_intervals(gain,features_data[0]['times'][mask],metadata['start'],metadata['end']):
                    intervals.append(dict(region=region,variant=variant,reference=name,**row,positive_favors='new likelihood-only magnitude head'))
    pd.DataFrame(scores).to_csv(report/'seed_scores.csv',index=False)
    summary=pd.DataFrame(scores).groupby(['region','variant','model']).agg(nll_mean=('nll_event','mean'),nll_fit_sd=('nll_event','std'),fits=('seed','size'),events=('n','first')).reset_index()
    summary.to_csv(report/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(report/'paired_block_intervals.csv',index=False)
    pd.DataFrame(pits).to_csv(report/'randomized_pit_histograms.csv',index=False)
    pd.DataFrame(briers).to_csv(report/'tail_brier_scores.csv',index=False)
    (report/'reference_selection.json').write_text(json.dumps(reference_settings,indent=2)+'\n')
    (report/'status.json').write_text(json.dumps(dict(state='completed',selection_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        target='proper probability of reported magnitude bin; conditional on observed history; same frozen encoder/time head',
        scope='component probability comparison; full autoregressive count/max calibration not claimed'),indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)
    print(pd.DataFrame(intervals).query('block_segments == 4').to_string(index=False),flush=True)


if __name__=='__main__':
    main()
