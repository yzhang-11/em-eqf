"""Summarize every test window, keeping fit and temporal dependence explicit."""
import json
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.utils.catalog_tests import run_number_test_result
BASE=ROOT/'experiments/revision/main/reports/calibration'
OUT=BASE/'analysis'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--region',choices=['swc','scsn'])
    selected_region=parser.parse_args().region
    regions=['swc','scsn'] if selected_region is None else [selected_region]
    out=OUT if selected_region is None else BASE/f'analysis_{selected_region}'
    out.mkdir(exist_ok=True)
    while True:
        try:
            state=json.loads((BASE/'status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(1)
            continue
        if state.get('state')=='failed':
            raise RuntimeError('Calibration failed; no partial-fit aggregation')
        ready=False
        if selected_region:
            try:
                snapshot=pd.read_csv(BASE/'window_scores.csv')
                snapshot=snapshot[snapshot.region==selected_region]
                ready=set(snapshot.seed)==set(range(5)) and all((BASE/f'{selected_region}_seed_{seed}.npz').exists() for seed in range(5))
            except (FileNotFoundError,pd.errors.EmptyDataError):
                pass
        if state.get('state')=='completed' or ready:
            break
        time.sleep(30)
    frame=pd.read_csv(BASE/'window_scores.csv')
    frame=frame[frame.region.isin(regions)]
    plan=json.loads((BASE/'plan.json').read_text())
    assert plan['fit_seeds']==list(range(5))
    fit_rows=[];interval_rows=[];rank_rows=[];number_rows=[]
    metric_columns=['count_CRPS','no_event_Brier','predicted_count','count_p99','N_test_reject_95']
    metric_columns += [c for c in frame if '_coverage_' in c or '_width_' in c]
    for region in regions:
        part=frame[frame.region==region]
        common=None
        for seed in range(5):
            rows=part[part.seed==seed].sort_values('window')
            observed=rows[['start','end','observed_count','observed_maximum']].to_numpy()
            if common is None:
                common=observed
            else:
                np.testing.assert_allclose(observed,common,rtol=0,atol=1e-8,equal_nan=True)
            assert rows.window.tolist()==list(range(len(rows)))
            np.testing.assert_allclose(rows.end.to_numpy()[:-1],rows.start.to_numpy()[1:],rtol=0,atol=1e-8)
            with np.load(BASE/f'{region}_seed_{seed}.npz') as file:
                np.testing.assert_array_equal(file['observed_count'],rows.observed_count)
                assert file['sim_count'].shape==(len(rows),512)
                assert int(file['sim_count'].max())<plan['sampler_buffer_events']
                rejection=[]
                for index,counts in enumerate(file['sim_count']):
                    result=run_number_test_result(counts,int(file['observed_count'][index]),
                        min_mw=3. if region=='swc' else 2.5,plot=False)
                    low_tail,high_tail=map(float,result.quantile)
                    rejected=low_tail<.025 or high_tail<.025
                    rejection.append(rejected)
                    number_rows.append(dict(region=region,seed=seed,window=index,
                        observed_count=int(file['observed_count'][index]),
                        partial_window=bool(rows.iloc[index].partial_window),
                        delta_1=low_tail,delta_2=high_tail,reject_central_95=rejected))
                rows=rows.copy();rows['N_test_reject_95']=rejection
            for scope,selected in [('all_windows',rows),('full_30day',rows[~rows.partial_window]),('final_partial',rows[rows.partial_window])]:
                row=dict(region=region,seed=seed,scope=scope,windows=len(selected),
                    observed_empty_windows=int((selected.observed_count==0).sum()),
                    observed_count_mean=float(selected.observed_count.mean()),
                    maximum_sim_count=int(selected.maximum_sim_count.max()))
                for metric in metric_columns:
                    row[metric]=float(pd.to_numeric(selected[metric].replace({'True':1.,'False':0.})).mean())
                fit_rows.append(row)
            for measure in ['count_rank','maximum_state_rank']:
                values=rows.loc[~rows.partial_window,measure].to_numpy()
                assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
                counts,_=np.histogram(values,bins=np.linspace(0,1,11))
                for index,count in enumerate(counts):
                    rank_rows.append(dict(region=region,seed=seed,measure=measure,bin=index,
                        lower=index/10,upper=(index+1)/10,windows=int(count)))
        # One temporal window is shared by five fitted models; it is not five
        # independent observations. Resample its index jointly across fits.
        full=part[~part.partial_window]
        n=len(full[full.seed==0])
        for measure in ['count','maximum_state']:
            for nominal in [.8,.9,.95]:
                column=f'{measure}_coverage_{round(nominal*100)}'
                values=np.asarray([full[full.seed==seed].sort_values('window')[column].to_numpy(dtype=float) for seed in range(5)])
                assert values.shape==(5,n) and np.isfinite(values).all()
                for block in [1,2,4,8]:
                    rng=np.random.default_rng(20261102)
                    starts=rng.integers(0,n,size=(2000,int(np.ceil(n/block))))
                    indices=((starts[:,:,None]+np.arange(block))%n).reshape(2000,-1)[:,:n]
                    fits=rng.integers(0,5,size=(2000,5))
                    sampled=values[fits[:,:,None],indices[:,None,:]].mean(axis=(1,2))
                    low,high=np.quantile(sampled,[.025,.975])
                    interval_rows.append(dict(region=region,measure=measure,nominal=nominal,
                        observed_coverage=float(values.mean()),low95=float(low),high95=float(high),
                        block_windows=block,block_days=block*30,effective_blocks=n/block,
                        unique_full_windows=n,fit_seeds=5,resamples=2000,
                        scope='conditional on this catalog/settings; fitted seeds and shared calendar blocks resampled; not independent-binomial CI'))
    fit_frame=pd.DataFrame(fit_rows)
    fit_frame.to_csv(out/'fit_scores.csv',index=False)
    numeric=[c for c in fit_frame if c not in ['region','seed','scope']]
    grouped=fit_frame.groupby(['region','scope'])[numeric].agg(['mean','std']).reset_index()
    grouped.columns=['_'.join(str(x) for x in c if x).rstrip('_') if isinstance(c,tuple) else c for c in grouped.columns]
    grouped.to_csv(out/'fit_variation_summary.csv',index=False)
    pd.DataFrame(interval_rows).to_csv(out/'coverage_block_intervals.csv',index=False)
    pd.DataFrame(rank_rows).to_csv(out/'randomized_rank_histograms.csv',index=False)
    pd.DataFrame(number_rows).to_csv(out/'number_test_windows.csv',index=False)
    (out/'status.json').write_text(json.dumps(dict(state='completed',fitted_models=5*len(regions),
        fit_windows=len(frame),unique_windows=len(frame[frame.seed==0]),
        scope='count and maximum-state calibration; magnitude-histogram M-test remains pending'),indent=2))
    print(grouped.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
