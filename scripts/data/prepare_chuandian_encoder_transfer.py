"""Shared purged targets and past-only features for all downstream baselines."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT/'scripts/tuning'))
import mamba3_transfer_common as c

BASE=ROOT/'experiments/revision/controls'
REPORTS=BASE/'reports/chuandian_multiregion_transfer_20261105'
DATA=c.STORE/'chuandian_transfer.pt'
DATES=dict(train_end='2011-02-12',validation_end='2016-04-03',test_end='2021-05-24')
FEATURES=['Num','Mag_max','Mag_mean','b_lsq','a_lsq','b_std_lsq','std_gr_lsq',
          'b_mlk','a_mlk','b_std_mlk','std_gr_mlk','dM_lsq','dM_mlk','Energy_sqrt',
          'prob_x7_lsq','prob_x7_mlk','beta','zvalue']


def main():
    assert not DATA.exists(),'Prepared study data are immutable'
    torch.set_num_threads(1);c.bootstrap()
    from src.data.event_pipeline import load_event_windows_for_task,build_full_window_feature_frame,build_inner_sliding_feature_tensor
    from src.data.preprocessing import load_and_filter_catalog
    from src.data.preparation_modules.common import _target_period_indices
    from src.data.event_loader import EventDataset
    catalog=ROOT/'data/ChuanDian';REPORTS.mkdir(parents=True,exist_ok=True)
    train_end=(pd.Timestamp(DATES['train_end'])-pd.Timestamp('1970-01-01')).total_seconds()/86400
    df=load_and_filter_catalog(catalog,Mc=3.)
    train=df.loc[df.t<train_end]
    excess=(train.Magnitude.to_numpy()-3.)/.1
    b=float(np.log1p(1/max(float(excess.mean()),1e-9)))/(.1*np.log(10))
    gaps=np.diff(train.t.to_numpy())
    metadata=[3.,b,float(gaps.mean()),float(gaps.min()),float(gaps.max()),.1,0]
    tasks={};audit={}
    for task,tw,tf,mf in [('classification',180,30,4.5),('regression',360,180,3.)]:
        bundle=load_event_windows_for_task(catalog,task_type=task,Mc=3.,Mf=mf,Twindow=tw,Tfore=tf,
            dt=10,context_len=0,fit_end=train_end,cache_namespace='review_v2')
        origins=np.array([s['t'] for s in bundle.samples_list])
        target_set=EventDataset(bundle.array_dict,mf,task_type=task,mag_min=3.,mag_max=9.,forecast_origins=origins)
        # Map through the existing target-validity rule (e.g. no future-event
        # maximum is defined for an empty regression target).
        kept=np.searchsorted(origins,target_set.forecast_origins)
        nonempty=np.array([len(bundle.array_dict['history']['t'][i])>0 for i in kept])
        kept=kept[nonempty];labels=np.asarray(target_set.labels,dtype=np.float32)[nonempty]
        selected_origins=origins[kept]
        args=OmegaConf.create(dict(split_dates=DATES,Tfore=tf,split_by_time=True,save_dir=str(REPORTS/task)))
        split_indices=_target_period_indices(selected_origins,args)
        rf=build_full_window_feature_frame(bundle.array_dict,bundle.samples_list,FEATURES,
            Mc=3.,dMag=.1,t_elaps_mode='window',change_rate_twindow=float(tw)).iloc[kept][FEATURES].to_numpy()
        lstm,_,_=build_inner_sliding_feature_tensor(bundle.array_dict,bundle.samples_list,FEATURES,
            Mc=3.,dMag=.1,t_elaps_mode='window',change_rate_twindow=float(tw),Twindow=tw,
            dt=10,time_step=5,feature_window=tw-40,feature_step=10)
        lstm=lstm[kept]
        scalers={}
        for name,values in [('rf',rf),('lstm_features',lstm)]:
            values=np.where(np.isfinite(values),values,np.nan)
            fit=values[split_indices[0]].reshape(-1,len(FEATURES))
            imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(fit)
            flat=imp.transform(values.reshape(-1,len(FEATURES)))
            scale=StandardScaler().fit(imp.transform(fit))
            normalized=scale.transform(flat).reshape(values.shape).astype(np.float32)
            assert np.isfinite(normalized).all()
            scalers[name]=dict(imputer_statistics=imp.statistics_.tolist(),mean=scale.mean_.tolist(),scale=scale.scale_.tolist())
            if name=='rf':rf=normalized
            else:lstm=normalized
        histories=[]
        for index in kept:
            hist=bundle.array_dict['history'];t=np.asarray(hist['t'][index],dtype=np.float64)
            mag=np.asarray(hist['Magnitude'][index],dtype=np.float64)
            dt=np.asarray(hist['dt'][index],dtype=np.float64)
            assert t.max()<origins[index] and t.min()>=origins[index]-tw
            # Same feature order as RegionalBatchAdapter: sorted keys.
            x=np.column_stack([np.log(np.maximum(dt,1e-10))-np.log(metadata[2]),
                               (mag-3.)*b-1/np.log(10)]).astype(np.float32)
            times=(t-(origins[index]-tw))/metadata[2]
            relative=((t-(origins[index]-tw))/tw).astype(np.float32)
            histories.append(dict(features=x,times=times,relative_time=relative))
        tasks[task]=dict(histories=histories,labels=labels,origins=selected_origins,
            splits=dict(zip(['train','validation','test'],[np.asarray(v) for v in split_indices])),
            rf=rf,lstm_features=lstm,scalers=scalers,Twindow=tw,Tfore=tf,Mf=mf)
        audit[task]=dict(Twindow=tw,Tfore=tf,Mf=mf,Mc=3.,dt=10,context_len=0,
            counts={k:len(v) for k,v in tasks[task]['splits'].items()},
            label_means={k:float(labels[v].mean()) for k,v in tasks[task]['splits'].items()},
            target_origin_sha256={k:hashlib.sha256(selected_origins[v].tobytes()).hexdigest() for k,v in tasks[task]['splits'].items()},
            history_min=int(min(len(h['times']) for h in histories)),history_max=int(max(len(h['times']) for h in histories)),
            history_mean=float(np.mean([len(h['times']) for h in histories])),
            empty_history_removed=int((~nonempty).sum()),feature_columns=FEATURES,
            lstm_inner_windows=5,lstm_feature_window=tw-40,lstm_feature_step=10)
        print(task,json.dumps(audit[task]),flush=True)
    source=catalog/'raw/chuandian_2021.csv'
    payload=dict(schema='shared_chuandian_encoder_transfer_v1',tasks=tasks,metadata=metadata,
        split_dates=DATES,source=str(source),source_sha256=c.sha(source),
        normalization='Target event statistics and engineered feature imputers/scalers use target training period only.',
        input_feature_order=['log_inter_times','mag'],label_range=[3.,9.])
    torch.save(payload,DATA)
    target=catalog/'catalogs/chuandian_multiregion_transfer_20261105.pt';target.parent.mkdir(exist_ok=True,parents=True)
    assert not target.exists();target.symlink_to(DATA)
    c.save(REPORTS/'data.json',dict(path=str(target),storage=str(DATA),sha256=c.sha(DATA),source=str(source),
        source_sha256=c.sha(source),split_dates=DATES,metadata=metadata,tasks=audit,
        target_definition='Classification: >=4.5 event in next30days. Regression: maximum magnitude in next180days, observed nonempty targets.',
        normalization=payload['normalization'],same_targets_all_methods=True))
    print('CHUANDIAN_DATA_READY',DATA,flush=True)


if __name__=='__main__':main()
