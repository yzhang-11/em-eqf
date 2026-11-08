"""Frozen GAS evaluation on reused 2020-24 and first-read 2025 SCEDC marks."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/analyze'))
from src.data import Sequence,Batch,TppDataset
from src.data.preprocessing import resolve_tied_times
from src.models.tpp.magnitude_dynamics import filter_gas
from src.train.config_setup import load_and_prepare_model
from compare_magnitude_models import block_intervals,gr_logp
from evaluate_magnitude_external import annual

BASE=ROOT/'experiments/revision/magnitude_models'
OUT=BASE/'reports/dynamics/external'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    choice_path=BASE/'reports/dynamics/selection.json'
    selected=json.loads(choice_path.read_text())
    assert selected['choices']['scsn']=='gr_free'
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=digest(choice_path),
        periods={'2020_2024':['2020-01-01','2025-01-01'],'2025':['2025-01-01','2026-01-01']},
        first_read_confirmation='2025 annual catalog downloaded only after GAS family and training fits frozen',
        reused_period='2020-2024 already inspected in prior modeling; exploratory continuation',
        fitting='none on either external period; no adaptation of static coefficients or hyperparameters',
        filtering='causal state updates after an observed mark; current mark excluded from own predictive state',
        inference='original neural baseline: common capacity 65536, fixed known pre-2020 XPOS center; no time-code edits',
        references='five frozen original NN-GR fits and train-only static GR; candidate is one deterministic fitted model')
    path=OUT/'plan.json'
    if path.exists():assert json.loads(path.read_text())['selection_sha256']==plan['selection_sha256']
    else:save(path,plan)
    frame,manifest=annual(2025)
    save(OUT/'download_2025.json',manifest)
    frame=frame[(frame.ET=='eq')&(frame.ts>='2025-01-01')&(frame.ts<'2026-01-01')]
    old=pd.read_csv(ROOT/'data/SCEDC/raw/scsn_eq_only_snapshot_20261102.csv')
    middle=pd.read_csv(ROOT/'data/SCEDC/processed/scsn_eq_2020_2024_snapshot_20261104.csv')
    for value in [old,middle]:value['ts']=pd.to_datetime(value.ts,format='mixed')
    combined=pd.concat([old,middle,frame],ignore_index=True).sort_values('ts',kind='stable')
    epoch=pd.Timestamp('1981-01-01');end=float((pd.Timestamp('2026-01-01')-epoch)/pd.Timedelta(days=1))
    scores=[];intervals=[];proof=[]
    for variant,mc,cache in [('main',2.5,'25a12469'),('sensitivity',3.,'673adf4b')]:
        row=next(r for r in selected['best'] if r['region']=='scsn' and r['variant']==variant and r['family']=='gr_free')
        data=combined[combined.Magnitude>=mc]
        marks=data.Magnitude.to_numpy(dtype=np.float32)
        times=((data.ts-epoch)/pd.Timedelta(days=1)).to_numpy(dtype=float)
        times=resolve_tied_times(times,.01/86400,end)
        bins=np.rint((marks.astype(float)-mc)/.01).astype(int)
        assert np.max(abs(marks-(mc+.01*bins)))<2e-6
        reference=TppDataset.load_from_disk(ROOT/'data/SCEDC/catalogs'/cache/'full_sequence.pt')[0]
        np.testing.assert_array_equal(marks[:len(reference.mag)],reference.mag.numpy())
        np.testing.assert_allclose(times[:len(reference.mag)],reference.arrival_times.numpy(),atol=1e-7,rtol=0)
        gas=filter_gas(row['theta'],'gr_free',bins*.01,.01,10.-mc,return_trace=True)
        sequence=Sequence(inter_times=torch.tensor(np.diff(times,prepend=0.,append=end),dtype=torch.float64),
            mag=torch.tensor(marks),t_start=0.,t_nll_start=14244.).double()
        capacity=65536;assert len(sequence.inter_times)<=capacity
        batch=Batch.from_list([sequence],max_sample_len=capacity-len(sequence.inter_times)).to('cuda:0')
        neural=[]
        for seed in range(5):
            checkpoint=ROOT/'experiments/revision/main/runs'/f'generation_scsn_{variant}_seed_{seed}'/'best_model_1.pth'
            model,_=load_and_prepare_model(checkpoint,torch.device('cuda:0'),compile=False)
            ref=reference.double();ref.t_nll_start=12053.
            with torch.inference_mode():
                context_reference=model.get_context(Batch.from_list([ref],max_sample_len=capacity-len(ref.inter_times)).to('cuda:0'))[0,:len(reference.mag)].cpu().numpy().copy()
            for rotary in model.modules():
                if rotary.__class__.__name__!='RotaryEmbeddingTime':continue
                forward=rotary.forward
                def fixed_forward(self,*args,_forward=forward,**kwargs):
                    kwargs['seqlen_offset']=1
                    return _forward(*args,**kwargs)
                rotary.forward=types.MethodType(fixed_forward,rotary)
            with torch.inference_mode():
                context=model.get_context(batch)
                b=model._get_b_pred(context,predict_b=True)[0,:len(marks)].double().cpu().numpy()
            prefix_error=float(np.max(abs(context[0,:len(reference.mag)].cpu().numpy()-context_reference)))
            assert prefix_error<1e-6,(variant,seed,prefix_error)
            values=gr_logp(b*np.log(10.),bins*.01,.01,10.-mc)
            neural.append(values)
            earlier=(times>=14244.)&(times<16071.)
            with np.load(BASE/'reports/external_2020_2024'/f'{variant}_seed_{seed}.npz') as previous:
                np.testing.assert_array_equal(marks[earlier],previous['magnitude'])
                np.testing.assert_array_equal(times[earlier],previous['times'])
                differences=values[earlier]-previous['original_logp']
                old_error=float(np.max(abs(differences)))
                old_mean_absolute_error=float(np.mean(abs(differences)))
                # Identical events/weights can differ slightly under the existing
                # accelerated encoder when future rows change. Audit both the
                # per-event maximum and average effect on the reported metric.
                assert old_error<1e-3 and old_mean_absolute_error<1e-5
            extension_error=None
            if seed==0:
                keep=times<16071.
                shorter=Sequence(inter_times=torch.tensor(np.diff(times[keep],prepend=0.,append=16071.),dtype=torch.float64),
                    mag=torch.tensor(marks[keep]),t_start=0.,t_nll_start=14244.).double()
                with torch.inference_mode():
                    prefix_context=model.get_context(Batch.from_list([shorter],max_sample_len=capacity-len(shorter.inter_times)).to('cuda:0'))
                    prefix_b=model._get_b_pred(prefix_context,predict_b=True)[0,:keep.sum()].double().cpu().numpy()
                prefix_values=gr_logp(prefix_b*np.log(10.),bins[keep]*.01,.01,10.-mc)
                extension_error=float(np.max(abs(prefix_values-values[keep])))
                assert extension_error<1e-3
            proof.append(dict(variant=variant,seed=seed,prefix_error=prefix_error,
                reused_2020_2024_original_logp_error=old_error,
                reused_2020_2024_mean_absolute_logp_error=old_mean_absolute_error,
                same_run_extension_logp_error=extension_error,checkpoint_sha256=digest(checkpoint)))
            print(json.dumps(proof[-1]),flush=True)
            del model,context;torch.cuda.empty_cache()
        neural=np.asarray(neural)
        with np.load(BASE/'features'/f'scsn_{variant}_events.npz') as f:
            x=f['bins'][f['split']==1]*.01
        static=minimize_scalar(lambda rate:-gr_logp(rate,x,.01,10.-mc).mean(),
            bounds=(.2*np.log(10.),2.*np.log(10.)),method='bounded').x
        static_values=gr_logp(static,bins*.01,.01,10.-mc)
        for name,(a,b) in plan['periods'].items():
            start=float((pd.Timestamp(a)-epoch)/pd.Timedelta(days=1));finish=float((pd.Timestamp(b)-epoch)/pd.Timedelta(days=1))
            mask=(times>=start)&(times<finish)
            candidates=[('GR_GAS',0,gas['logp']),('static_GR',0,static_values)]
            candidates += [('original_GR',seed,values) for seed,values in enumerate(neural)]
            for label,seed,values in candidates:
                scores.append(dict(period=name,variant=variant,Mc=mc,model=label,fit=seed,events=int(mask.sum()),nll_event=float(-values[mask].mean())))
            for comparison,reference_values in [('original_GR',neural[:,mask]),('static_GR',static_values[mask][None,:])]:
                gains=gas['logp'][mask][None,:]-reference_values
                for item in block_intervals(gains,times[mask],start,finish):
                    intervals.append(dict(period=name,variant=variant,Mc=mc,reference=comparison,**item))
            np.savez_compressed(OUT/f'{variant}_{name}.npz',times=times[mask],magnitude=marks[mask],
                gas_logp=gas['logp'][mask],gas_b=gas['value'][mask],original_logp=neural[:,mask],static_logp=static_values[mask])
        del batch,sequence
    pd.DataFrame(scores).to_csv(OUT/'fit_scores.csv',index=False)
    summary=pd.DataFrame(scores).groupby(['period','variant','Mc','model']).agg(nll_mean=('nll_event','mean'),events=('events','first')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'paired_block_intervals.csv',index=False)
    save(OUT/'status.json',dict(state='completed',proof=proof,created_utc=datetime.now(timezone.utc).isoformat()))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
