"""Full-test count and maximum-state diagnostics using the existing sampler."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.catalogs.pathing import build_tpp_catalog_init_kwargs
from src.data import Catalog
from src.train.config_setup import load_and_prepare_model
from src.utils.forecast_eval import (
    SlidingWindowEvaluationRange, SlidingWindowForecastConfig,
    _build_sliding_window_bounds, compute_mean_crps, run_sliding_window_forecast,
)

BASE = ROOT/'experiments/revision/main'


class HistorySampler:
    """Remove an irrelevant NLL cutoff from a pure observed-history sequence."""
    def __init__(self, model):
        self.model=model

    def __getattr__(self, name):
        return getattr(self.model,name)

    def sample(self, **kwargs):
        past=kwargs['past_seq']
        # The evaluator sets a late NLL cutoff. The sampler then takes a
        # zero-duration slice at the last event, before that cutoff. No loss
        # is evaluated on this history; event times, marks and origin stay fixed.
        past.t_nll_start=past.t_start
        total=kwargs.pop('batch_size')
        kwargs['max_sample_len']=16384
        forecasts=[]
        for offset in range(0,total,128):
            forecasts.extend(self.model.sample(batch_size=min(128,total-offset),**kwargs))
        return forecasts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--wait-controls',action='store_true')
    args=parser.parse_args()
    out=(ROOT/'experiments/revision/archive/calibration/calibration_cache_fix_pilot'
         if args.pilot else BASE/'reports/calibration')
    out.mkdir(parents=True,exist_ok=True)
    fix_manifest=ROOT/'experiments/revision/archive/development/attention_cache_fix_20261102/manifest.json'
    assert json.loads(fix_manifest.read_text())['validation_status']=='passed'
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
        'src/models/mha/mha_time.py','src/models/tpp/mixer_tpp.py','src/models/mha/rotary_embedding_time.py']}
    plan=dict(horizon_days=30,slide_days=30,include_final_partial_window=True,
        samples_per_window=32 if args.pilot else 512,fit_seeds=[0] if args.pilot else list(range(5)),
        levels=[.8,.9,.95],selection='all test windows; no selection on observed outcomes',
        history='all observed events before each forecast origin, including earlier test observations',
        empty_window='count=0; maximum state is a distinct ordered no-event atom below any magnitude',
        history_metadata='pure-history t_nll_start reset to t_start before sampling; no event/time changes',
        sampler_buffer_events=16384,sampling_chunk_size=128,cache_per_window=True,
        inference_source_sha256=source_hashes,
        sampling_seed='20261102 + 100000*region_index + 1000*fit_seed + window_index',
        scope='count and maximum-state calibration diagnostics; not a PyCSEP magnitude-distribution M-test',
        uncertainty='finite ensemble ranks and fit-to-fit summaries; no binomial independent-window CI')
    path=out/'plan.json'
    if path.exists():
        assert json.loads(path.read_text())==plan
    else:
        path.write_text(json.dumps(plan,indent=2))
    (out/'status.json').write_text(json.dumps(dict(state='queued' if args.wait_controls else 'running',
        pilot=args.pilot,created_utc=datetime.now(timezone.utc).isoformat()),indent=2))
    if args.wait_controls:
        while True:
            statuses=[json.loads(p.read_text()) for p in (ROOT/'experiments/revision/controls/runs').glob('*/job_status.json')]
            if any(s['state']=='failed' for s in statuses):
                raise RuntimeError('Control failure requires review before launching next GPU workload')
            if len(statuses)==60 and all(s['state']=='completed' for s in statuses):
                break
            time.sleep(30)
    (out/'status.json').write_text(json.dumps(dict(state='running',pilot=args.pilot,
        started_utc=datetime.now(timezone.utc).isoformat()),indent=2))
    rows=[]
    for region_index,region in enumerate(['swc','scsn']):
        config=OmegaConf.load(BASE/'configs'/f'generation_{region}_main_seed_0.yaml')
        cls=Catalog.by_name(config.dataset+'-Standard')
        cat=cls(**build_tpp_catalog_init_kwargs(cls,ROOT/'data'/config.dataset,config.catalog_cfg))
        seq=cat.test[0].double()
        start,end=float(seq.t_nll_start),float(seq.t_end)
        if args.pilot:
            end=min(start+60,end)
        evaluation_range=SlidingWindowEvaluationRange(name='test',start=start,end=end,grid_anchor=start)
        starts,ends=_build_sliding_window_bounds(seq,duration=30,slide_step=30,
            evaluation_range=evaluation_range,include_truncated_final_window=True)
        assert len(starts) and abs(starts[0]-start)<1e-8 and abs(ends[-1]-end)<1e-8
        np.testing.assert_allclose(ends[:-1],starts[1:],rtol=0,atol=1e-8)
        times=seq.arrival_times.numpy();magnitudes=seq.mag.numpy()
        # The legacy extractor includes endpoints; these grids have no endpoint events.
        assert not np.isin(np.concatenate([starts,ends]),times).any(), 'Endpoint event needs explicit half-open extraction'
        left=np.searchsorted(times,starts,side='left');right=np.searchsorted(times,ends,side='left')
        counts=right-left
        maxima=np.array([magnitudes[a:b].max() if b>a else np.nan for a,b in zip(left,right)])
        assert counts.sum()==np.count_nonzero((times>=start)&(times<end))
        for seed in plan['fit_seeds']:
            torch.manual_seed(20261102+seed)
            np.random.seed(20261102+seed)
            checkpoint=BASE/'runs'/f'generation_{region}_main_seed_{seed}'/'best_model_1.pth'
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            cache=out/f'{region}_seed_{seed}.npz'
            if cache.exists():
                with np.load(cache) as file:
                    assert str(file['checkpoint_sha256'])==digest
                    simulations=file['sim_count'].copy();marks=file['sim_maximum'].copy()
                    np.testing.assert_allclose(file['start'],starts,rtol=0,atol=1e-8)
            else:
                model,_=load_and_prepare_model(checkpoint,torch.device('cuda:0'),compile=False)
                count_rows=[];mark_rows=[]
                window_folder=out/'windows'/f'{region}_seed_{seed}'
                window_folder.mkdir(parents=True,exist_ok=True)
                for window,(a,b) in enumerate(zip(starts,ends)):
                    sampling_seed=20261102+100000*region_index+1000*seed+window
                    window_cache=window_folder/f'{window:04d}.npz'
                    progress=dict(state='running',pilot=args.pilot,region=region,seed=seed,
                        window=window,windows_in_region=len(starts),start=float(a),end=float(b))
                    (out/'status.json').write_text(json.dumps(progress,indent=2))
                    if window_cache.exists():
                        with np.load(window_cache) as file:
                            assert str(file['checkpoint_sha256'])==digest and int(file['sampling_seed'])==sampling_seed
                            assert int(file['observed_count'])==counts[window]
                            assert float(file['start'])==a and float(file['end'])==b
                            c=file['sim_count'].copy();m=file['sim_maximum'].copy()
                    else:
                        print(region,seed,'window',window,'start',a,'sampling',flush=True)
                        torch.manual_seed(sampling_seed)
                        np.random.seed(sampling_seed)
                        result=run_sliding_window_forecast(HistorySampler(model),seq,torch.device('cuda:0'),
                            config=SlidingWindowForecastConfig(duration=30,slide_step=30,
                                samples_per_batch=plan['samples_per_window'],
                                evaluation_range=SlidingWindowEvaluationRange(name='test',start=a,end=b,grid_anchor=a),
                                include_truncated_final_window=True,return_sim_count_matrix=True,
                                compute_mag_max=True,return_sim_mag_max_matrix=True,
                                precomputed_counts=counts[window:window+1],precomputed_mag_max=maxima[window:window+1]))
                        np.testing.assert_array_equal(result.counts,counts[window:window+1])
                        np.testing.assert_allclose(result.mag_max,maxima[window:window+1],rtol=0,atol=1e-6,equal_nan=True)
                        c=result.sim_count_matrix[0];m=result.sim_mag_max_matrix[0]
                        np.savez_compressed(window_cache,start=a,end=b,observed_count=counts[window],
                            observed_maximum=maxima[window],sim_count=c,sim_maximum=m,
                            checkpoint_sha256=digest,sampling_seed=sampling_seed)
                        print(region,seed,'window',window,'max_sim_count',int(c.max()),'completed',flush=True)
                        del result
                    assert len(c)==len(m)==plan['samples_per_window']
                    count_rows.append(c);mark_rows.append(m)
                simulations=np.asarray(count_rows);marks=np.asarray(mark_rows)
                assert (simulations>=0).all() and np.isfinite(simulations).all()
                assert np.array_equal(np.isnan(marks),simulations==0)
                assert np.isfinite(marks[simulations>0]).all()
                np.savez_compressed(cache,start=starts,end=ends,observed_count=counts,observed_maximum=maxima,
                    sim_count=simulations,sim_maximum=marks,checkpoint_sha256=digest)
                del model
                torch.cuda.empty_cache()
            rng=np.random.default_rng(20261102+seed)
            states=np.where(simulations>0,marks,-np.inf)
            observed_states=np.where(counts>0,maxima,-np.inf)
            for window,(a,b) in enumerate(zip(starts,ends)):
                c=simulations[window];m=marks[window];state=states[window];observed=observed_states[window]
                valid=m[np.isfinite(m)]
                row=dict(region=region,seed=seed,window=window,start=a,end=b,duration=b-a,
                    partial_window=b-a<30-1e-8,observed_count=int(counts[window]),
                    observed_maximum=maxima[window],predicted_count=float(c.mean()),
                    maximum_sim_count=int(c.max()),count_p99=float(np.quantile(c,.99)),
                    no_event_probability=float(np.mean(c==0)),
                    no_event_Brier=float((np.mean(c==0)-(counts[window]==0))**2),
                    count_CRPS=compute_mean_crps(counts[window:window+1],simulations[window:window+1])[0],
                    count_rank=float((np.count_nonzero(c<counts[window])+rng.random()*(np.count_nonzero(c==counts[window])+1))/(len(c)+1)),
                    maximum_state_rank=float((np.count_nonzero(state<observed)+rng.random()*(np.count_nonzero(state==observed)+1))/(len(state)+1)))
                for level in plan['levels']:
                    alpha=(1-level)/2;key=str(round(level*100))
                    low,high=np.quantile(c,[alpha,1-alpha],method='inverted_cdf')
                    row['count_coverage_'+key]=bool(low<=counts[window]<=high)
                    row['count_width_'+key]=float(high-low)
                    low,high=np.quantile(state,[alpha,1-alpha],method='inverted_cdf')
                    row['maximum_state_coverage_'+key]=bool(low<=observed<=high)
                    if len(valid):
                        low,high=np.quantile(valid,[alpha,1-alpha],method='inverted_cdf')
                        row['conditional_maximum_width_'+key]=float(high-low)
                        row['conditional_maximum_coverage_'+key]=bool(low<=maxima[window]<=high) if counts[window]>0 else None
                rows.append(row)
            pd.DataFrame(rows).to_csv(out/'window_scores.csv',index=False)
            print(region,seed,'windows',len(starts),'sampling and scoring complete',flush=True)
    frame=pd.DataFrame(rows)
    keys=['count_CRPS','no_event_Brier','count_rank','maximum_state_rank']
    keys += [c for c in frame if any(tag in c for tag in ['_coverage_','_width_'])]
    summary=frame.groupby(['region','seed','partial_window']).agg(windows=('window','count'),
        observed_empty_windows=('observed_count',lambda x:int((x==0).sum())),
        **{key:(key,'mean') for key in keys}).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    (out/'status.json').write_text(json.dumps(dict(state='completed',pilot=args.pilot,
        completed_utc=datetime.now(timezone.utc).isoformat(),fit_windows=len(frame)),indent=2))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        folder=BASE/('calibration_cache_fix_pilot' if '--pilot' in sys.argv else 'calibration')
        status=folder/'status.json'
        if status.exists():
            payload=json.loads(status.read_text())
            payload.update(state='failed',error=str(error),failed_utc=datetime.now(timezone.utc).isoformat())
            status.write_text(json.dumps(payload,indent=2))
        raise
