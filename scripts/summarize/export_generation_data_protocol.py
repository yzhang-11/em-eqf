"""Export actual four-period dates and event counts for main and higher Mc."""
import hashlib
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.data import Catalog
from src.catalogs.pathing import build_tpp_catalog_init_kwargs
from src.utils.forecast_eval import resolve_catalog_time_reference


def main():
    rows=[]
    for region in ['swc','scsn']:
        for variant in ['main','sensitivity']:
            path=ROOT/'experiments/revision/main/configs'/f'generation_{region}_{variant}_seed_0.yaml'
            config=OmegaConf.load(path)
            cls=Catalog.by_name(config.dataset+'-Standard')
            cat=cls(**build_tpp_catalog_init_kwargs(cls,ROOT/'data'/config.dataset,config.catalog_cfg))
            seq=cat.full_sequence.double()
            times=seq.arrival_times.numpy()
            base,step=resolve_catalog_time_reference(cat)
            assert step==pd.Timedelta(days=1)
            boundaries=[seq.t_start,cat.train[0].t_nll_start,cat.val[0].t_nll_start,cat.test[0].t_nll_start,seq.t_end]
            expected=[len(times)-sum(part[0].num_nll_events for part in [cat.train,cat.val,cat.test])]+[part[0].num_nll_events for part in [cat.train,cat.val,cat.test]]
            source=Path(config.catalog_cfg.catalog_file)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            total=0
            for i,name in enumerate(['warmup','train','validation','test']):
                left,right=map(float,boundaries[i:i+2])
                count=int(np.count_nonzero((times>=left)&(times<right)))
                assert count==expected[i],(region,variant,name,count,expected[i])
                rows.append(dict(region=region,variant=variant,Mc=float(config.Mc),period=name,
                    start=str(base+left*step),end_exclusive=str(base+right*step),events=count,
                    duration_days=right-left,relative_start=left,relative_end_exclusive=right,
                    interval='[start,end)',source_file=str(source),source_sha256=digest,
                    likelihood_target=name!='warmup',
                    conditioning='all events observed before forecast origin; earlier test events allowed'))
                total+=count
            assert total==len(times)
    out=ROOT/'experiments/revision/generation_baselines/reports/generation_periods.csv'
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows)[['region','variant','Mc','period','start','end_exclusive','events']].to_string(index=False))


if __name__=='__main__':
    main()
