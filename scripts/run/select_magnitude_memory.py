"""Select an interpretable probability time constant using validation data only."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/run'))
from run_magnitude_models import OUT,load_data
from src.models.tpp.magnitude_density import MagnitudeDensityHead


def probability_matrix(saved,data,end,device='cuda:0'):
    config=saved['config'];mc,step=float(data['Mc']),float(data['step'])
    normalized=(data['features'][:end]-saved['normalization_mean'])/saved['normalization_std']
    model=MagnitudeDensityHead(saved['input_dim'],config['family'],config['components'],mc,step,hidden=32).to(device)
    model.load_state_dict(saved['state_dict']);model.eval()
    grid=torch.tensor(mc+np.arange(int(round((10.-mc)/step)))*step,device=device,dtype=torch.float64)
    result=np.empty((end,len(grid)),dtype=np.float64)
    with torch.inference_mode():
        for start in range(0,end,1024):
            x=torch.tensor(normalized[start:min(start+1024,end)],device=device)
            values=model(x).log_prob(grid.unsqueeze(-1)).exp().T.cpu().numpy()
            assert np.max(abs(values.sum(1)-1.))<1e-8
            result[start:start+len(values)]=values
    return result


def filter_probabilities(probabilities,times,tau):
    if tau==0:return probabilities.copy()
    result=np.empty_like(probabilities);state=probabilities[0].copy();result[0]=state
    for i in range(1,len(times)):
        alpha=-np.expm1(-(times[i]-times[i-1])/tau)
        state=(1.-alpha)*state+alpha*probabilities[i]
        result[i]=state
    return result


def main():
    torch.set_num_threads(1)
    selection=json.loads((OUT/'selection.json').read_text());taus=[0.,.1,1.,7.,30.]
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),time_constants_days=taus,
        update='p_i = exp(-dt_i/tau)*p_{i-1} + (1-exp(-dt_i/tau))*q_i(history)',
        bound='TV(p_i,p_{i-1}) <= 1-exp(-dt_i/tau) <= dt_i/tau',
        loss='no auxiliary penalties; frozen likelihood-only head',
        rule='minimum mean validation NLL across five fits and both Mc values, per region; no candidate test or external results used',
        selection_sha256=hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest())
    path=OUT/'configs/memory_plan.json'
    assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n')
    rows=[];choices={}
    for region in ['swc','scsn']:
        recipe=selection['recipes'][region]['selected']
        for variant in ['main','sensitivity']:
            for seed in range(5):
                name=f"confirm_{region}_{variant}_{recipe['family']}{recipe['components']}_{recipe['features']}_seed_{seed}"
                saved=torch.load(OUT/'runs'/name/'best_head.pth',map_location='cpu',weights_only=False)
                data=load_data(region,variant,seed,recipe['features'])
                end=np.flatnonzero(data['split']==2)[-1]+1;mask=data['split'][:end]==2
                probabilities=probability_matrix(saved,data,end)
                for tau in taus:
                    smoothed=filter_probabilities(probabilities,data['times'][:end],tau)
                    observed=smoothed[np.arange(end),data['bins'][:end]]
                    assert np.all(observed>0)
                    score=float(-np.log(observed[mask]).mean())
                    rows.append(dict(region=region,variant=variant,seed=seed,tau_days=tau,validation_nll=score))
                    del smoothed
                del probabilities
                pd.DataFrame(rows).to_csv(OUT/'reports/memory_validation_scores.csv',index=False)
        subset=pd.DataFrame([x for x in rows if x['region']==region]);means=subset.groupby('tau_days').validation_nll.mean()
        tau=float(means.idxmin());choices[region]={'tau_days':tau,'validation_nll':float(means.loc[tau]),'all_validation_means':means.to_dict()}
    (OUT/'configs/memory_selection.json').write_text(json.dumps(dict(state='validation_selected_before_smoothed_test_scoring',choices=choices),indent=2)+'\n')
    print(json.dumps(choices),flush=True)


if __name__=='__main__':main()
