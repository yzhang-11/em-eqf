"""Independent later-period confirmation of the already-frozen SCSN magnitude head."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import requests
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/analyze'))
from src.catalogs.scedc import COL_NAMES
from src.data import Sequence,Batch,TppDataset
from src.data.preprocessing import resolve_tied_times,estimate_mc_max_curvature
from src.models.tpp.magnitude_model import load_magnitude_model
from compare_magnitude_models import block_intervals,gr_logp

BASE=ROOT/'experiments/revision/magnitude_models'
OUT=BASE/'reports/external_2020_2024'
RAW=ROOT/'data/SCEDC/raw/scsn_annual_snapshot_20261104'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def annual(year):
    url=f'https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/{year}.catalog'
    path=RAW/f'{year}.catalog'
    if not path.exists():
        response=requests.get(url,timeout=(10,60));response.raise_for_status()
        assert '#YYY/MM/DD' in response.text[:5000]
        path.write_bytes(response.content)
    frame=pd.read_csv(path,sep=r'\s+',header=None,names=COL_NAMES,comment='#',index_col=False)
    frame['ts']=pd.to_datetime(frame['date'])+pd.to_timedelta(frame['time'])
    frame['Magnitude']=pd.to_numeric(frame.magnitude,errors='raise')
    return frame,dict(year=year,url=url,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(frame))


def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    selection_path=BASE/'selection.json';selected=json.loads(selection_path.read_text())['recipes']['scsn']['selected']
    recipe={k:selected[k] for k in ['family','components','features']}
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        years=list(range(2020,2025)),end_exclusive='2025-01-01',start='2020-01-01',
        primary='SCSN M>=2.5 selected magnitude head versus frozen original, proper bin probabilities',
        sensitivity='same selected head architecture at Mc=3.0; separately trained before reading external data',
        selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),recipe=recipe,
        models='five frozen fits; no fitting, recalibration, early stopping or hyperparameter changes using this period',
        uncertainty='paired fitted seeds and common circular calendar blocks; 20 segments, block 1/2/4, 2000 draws',
        conditioning='all frozen pre-2020 history plus earlier external observations; target mark excluded from own predictive features')
    plan['rotary_reference']='Pre-2020 observed-history reference center, fixed before external scoring; extending the sequence must preserve every old context.'
    plan['inference_capacity']=65536
    plan['capacity_reason']='Use identical tensor shapes for reference and extension; TF32 GEMM kernel selection must not change past contexts.'
    plan_path=OUT/'plan.json'
    if plan_path.exists():
        old=json.loads(plan_path.read_text());assert old['selection_sha256']==plan['selection_sha256']
        if 'rotary_reference' not in old:
            save(OUT/'plan_before_reference_center_check.json',old)
            old['rotary_reference']=plan['rotary_reference'];save(plan_path,old)
        if 'inference_capacity' not in old:
            save(OUT/'plan_before_fixed_capacity_check.json',old)
            old['inference_capacity']=plan['inference_capacity'];old['capacity_reason']=plan['capacity_reason'];save(plan_path,old)
    else:save(plan_path,plan)
    with ThreadPoolExecutor(max_workers=3) as pool:download=list(pool.map(annual,range(2020,2025)))
    save(OUT/'download_manifest.json',dict(downloaded_utc=datetime.now(timezone.utc).isoformat(),files=[r for _,r in download]))
    external=pd.concat([d for d,_ in download],ignore_index=True)
    external=external[(external.ET=='eq')&(external.ts>='2020-01-01')&(external.ts<'2025-01-01')].sort_values('ts',kind='stable')
    external.to_csv(ROOT/'data/SCEDC/processed/scsn_eq_2020_2024_snapshot_20261104.csv',index=False)
    diagnostics=[]
    for year in range(2020,2025):
        m=external[external.ts.dt.year==year].Magnitude.to_numpy()
        mc,*_=estimate_mc_max_curvature(m,bin_width=.1,plot=False)
        diagnostics.append(dict(year=year,events=len(m),MAXC=float(mc),n_ge_2p5=int((m>=2.5).sum()),n_ge_3=int((m>=3.).sum())))
    save(OUT/'catalog_quality.json',diagnostics)
    old=pd.read_csv(ROOT/'data/SCEDC/raw/scsn_eq_only_snapshot_20261102.csv')
    old['ts']=pd.to_datetime(old.ts,format='mixed')
    frame=pd.concat([old,external],ignore_index=True).sort_values('ts',kind='stable')
    epoch=pd.Timestamp('1981-01-01');start=float((pd.Timestamp('2020-01-01')-epoch)/pd.Timedelta(days=1));end=float((pd.Timestamp('2025-01-01')-epoch)/pd.Timedelta(days=1))
    all_scores=[];all_intervals=[];proof=[]
    for variant,mc,cache in [('main',2.5,'25a12469'),('sensitivity',3.,'673adf4b')]:
        filtered=frame[frame.Magnitude>=mc].copy()
        times=((filtered.ts-epoch)/pd.Timedelta(days=1)).to_numpy(dtype=float)
        times=resolve_tied_times(times,.01/86400,end)
        marks=filtered.Magnitude.to_numpy(dtype=np.float32)
        old_seq=TppDataset.load_from_disk(ROOT/'data/SCEDC/catalogs'/cache/'full_sequence.pt')[0]
        assert np.array_equal(marks[:len(old_seq.mag)],old_seq.mag.numpy())
        assert np.max(abs(times[:len(old_seq.mag)]-old_seq.arrival_times.numpy()))<1e-7
        sequence=Sequence(inter_times=torch.tensor(np.diff(times,prepend=0.,append=end),dtype=torch.float64),
                          mag=torch.tensor(marks,dtype=torch.float32),t_start=0.,t_nll_start=start).double()
        capacity=plan['inference_capacity']
        assert len(sequence.inter_times)<=capacity
        batch=Batch.from_list([sequence],max_sample_len=capacity-len(sequence.inter_times)).to('cuda:0');mask=times>=start
        newer=[];baseline=[]
        for seed in range(5):
            head=BASE/'runs'/f"confirm_scsn_{variant}_{recipe['family']}{recipe['components']}_{recipe['features']}_seed_{seed}"/'best_head.pth'
            model,_=load_magnitude_model(head,torch.device('cuda:0'))
            # Use the known history as the numerical XPOS reference. No fitted parameter changes.
            reference=old_seq.double()
            reference.t_nll_start=12053.
            with torch.inference_mode():
                reference_context=model.get_context(Batch.from_list([reference],max_sample_len=capacity-len(reference.inter_times)).to('cuda:0'))[0,:len(old_seq.mag)].cpu().numpy().copy()
            reference_centers=[]
            for rotary in model.modules():
                if rotary.__class__.__name__!='RotaryEmbeddingTime':continue
                reference_centers.append(float(rotary._center_cached))
                normal_forward=rotary.forward
                def fixed_reference_forward(self,*args,_forward=normal_forward,**kwargs):
                    kwargs['seqlen_offset']=1  # Skip center recomputation; explicit event times stay unchanged.
                    return _forward(*args,**kwargs)
                rotary.forward=types.MethodType(fixed_reference_forward,rotary)
            with torch.inference_mode():
                context=model.get_context(batch)
                logp=model.get_magnitude_dist(context).log_prob(batch.mag)[0,:len(marks)].cpu().numpy()[mask]
                b=model._get_b_pred(context,predict_b=True)[0,:len(marks)].double().cpu().numpy()[mask]
            with np.load(BASE/'features'/f'scsn_{variant}_seed_{seed}.npz') as d:
                cached_context_error=float(np.max(abs(reference_context-d['context'])))
            prefix_error=float(np.max(abs(context[0,:len(old_seq.mag)].cpu().numpy()-reference_context)))
            assert prefix_error<1e-6,(variant,seed,prefix_error)
            lower=np.rint((marks[mask].astype(float)-mc)/.01)*.01
            old_logp=gr_logp(b*np.log(10.),lower,.01,10.-mc)
            assert np.isfinite(logp).all() and np.isfinite(old_logp).all()
            newer.append(logp);baseline.append(old_logp)
            for label,values in [('original',old_logp),('selected',logp)]:
                all_scores.append(dict(variant=variant,Mc=mc,seed=seed,model=label,events=int(mask.sum()),nll_event=float(-values.mean())))
            np.savez_compressed(OUT/f'{variant}_seed_{seed}.npz',times=times[mask],magnitude=marks[mask],
                                selected_logp=logp,original_logp=old_logp)
            proof.append(dict(variant=variant,seed=seed,original_history_context_error=prefix_error,events=int(mask.sum()),
                              cross_run_cached_context_difference=cached_context_error,
                              reference_centers=reference_centers,
                              head_sha256=hashlib.sha256(head.read_bytes()).hexdigest()))
            print(json.dumps(proof[-1]),flush=True)
            del model,context;torch.cuda.empty_cache()
        for row in block_intervals(np.asarray(newer)-np.asarray(baseline),times[mask],start,end):
            all_intervals.append(dict(variant=variant,Mc=mc,reference='original',**row))
        del batch,sequence
    pd.DataFrame(all_scores).to_csv(OUT/'seed_scores.csv',index=False)
    pd.DataFrame(all_scores).groupby(['variant','Mc','model']).agg(nll_mean=('nll_event','mean'),nll_fit_sd=('nll_event','std'),events=('events','first')).to_csv(OUT/'summary.csv')
    pd.DataFrame(all_intervals).to_csv(OUT/'paired_block_intervals.csv',index=False)
    save(OUT/'status.json',dict(state='completed',proof=proof,updated_utc=datetime.now(timezone.utc).isoformat()))
    print(pd.DataFrame(all_intervals).to_string(index=False),flush=True)


if __name__=='__main__':main()
