"""Count old-rule exclusions and empty targets on the actual frozen windows."""
import copy
import json
import logging
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.data.event_loader import compute_classification_label
from src.data.event_pipeline import load_event_windows_for_task
from src.data.preparation import prepare_data
from src.data.preparation_modules.common import _target_period_indices, _window_options

BASE=ROOT/'experiments/revision/main'
OUT=BASE/'reports/window_protocol_audit'


def main():
    logging.basicConfig(level=logging.ERROR)
    OUT.mkdir(exist_ok=True)
    rows=[];intersections=[]
    names=[f'clf_azdx_h{h}_seed_0' for h in [10,20,30,60,90]]+['reg_azdx_seed_0']
    for name in names:
        config=OmegaConf.load(BASE/'configs'/f'{name}.yaml')
        config.save_dir=str(OUT/name)
        df,tr,va,te,ds=prepare_data(config,ROOT/'data'/config.dataset)
        bundle=load_event_windows_for_task(base_dir=ROOT/'data'/config.dataset,task_type=config.task_type,
            Mc=config.Mc,Mf=config.Mf,Twindow=config.Twindow,Tfore=config.Tfore,dt=config.dt,
            context_len=0,**_window_options(config))
        origins=np.array([sample['t'] for sample in bundle.samples_list])
        silent=copy.deepcopy(config);silent.save_dir=None
        indices=_target_period_indices(origins,silent)
        split=np.full(len(origins),'purged',dtype=object)
        for tag,index in zip(['train','validation','test'],indices):
            split[index]=tag
        future=bundle.array_dict['future']
        empty=np.array([len(value)==0 for value in future['t']])
        history_empty=np.array([len(value)==0 for value in bundle.array_dict['history']['t']])
        actual_origins=np.asarray(ds.forecast_origins)
        np.testing.assert_allclose(actual_origins,origins if config.task_type=='classification' else origins[~empty],rtol=0,atol=1e-8)
        current=old=None
        if config.task_type=='classification':
            current=np.array([compute_classification_label(value,[],config.Mf) for value in future['t']])
            np.testing.assert_array_equal(current,ds.labels)
            filtered=df[df.Magnitude>=config.Mc].sort_values('t',kind='stable')
            times=filtered.t.to_numpy();mag=filtered.Magnitude.to_numpy()
            old=np.array([compute_classification_label(value,
                mag[np.searchsorted(times,t-config.Tfore):np.searchsorted(times,t+config.Tfore)],config.Mf)
                for t,value in zip(origins,future['t'])])
            assert np.array_equal(old[current==1],current[current==1])
        for tag in ['all','train','validation','test','purged']:
            index=np.arange(len(origins)) if tag=='all' else np.flatnonzero(split==tag)
            row=dict(config=name,task=config.task_type,horizon=config.Tfore,Mf=config.Mf,Mc=config.Mc,
                split=tag,candidate_windows=len(index),empty_history_windows=int(history_empty[index].sum()),
                empty_future_windows=int(empty[index].sum()),
                empty_target_handling='retained as negative' if current is not None else 'excluded; no imputed magnitude')
            if current is not None:
                valid=np.isfinite(old[index])
                row.update(future_only_positive=int(current[index].sum()),future_only_negative=int((current[index]==0).sum()),
                    old_context1_retained=int(valid.sum()),old_context1_removed_negative=int((~valid).sum()),
                    future_only_positive_fraction=float(current[index].mean()) if len(index) else None,
                    old_context1_positive_fraction=float(old[index][valid].mean()) if valid.any() else None)
            rows.append(row)
        target_sets=[]
        for loader in [tr,va,te]:
            events=set()
            for origin in actual_origins[loader.dataset.indices]:
                events.update(df.index[(df.t>=origin)&(df.t<origin+config.Tfore)&(df.Magnitude>=config.Mf)].tolist())
            target_sets.append(events)
        sizes=[len(target_sets[a]&target_sets[b]) for a,b in [(0,1),(0,2),(1,2)]]
        assert sizes==[0,0,0]
        intersections.append(dict(config=name,train_validation=sizes[0],train_test=sizes[1],validation_test=sizes[2]))
    pd.DataFrame(rows).to_csv(OUT/'sample_composition.csv',index=False)
    (OUT/'target_event_intersections.json').write_text(json.dumps(intersections,indent=2))
    print(pd.DataFrame(rows).query('split=="test"').to_string(index=False))


if __name__=='__main__':
    main()
