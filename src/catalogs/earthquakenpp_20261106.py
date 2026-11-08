"""Author-frozen EarthquakeNPP catalogs; original experiments remain unchanged.

Published files provide UTC time, magnitude, geographic coordinates and projected
x/y, but no focal depth. These adapters keep depth absent rather than fill zeros.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import Catalog,Sequence,TppDataset,default_catalogs_dir
from src.catalogs.external_catalog import file_sha256,utc_naive
from src.data.preprocessing import resolve_tied_times
from src.utils.catalog_pathing import build_hashed_catalog_root
from src.utils.catalog_utils import train_val_test_split_sequence

COMMIT='26d18048e1ca8ff2b02c7016b993de48ed0760f5'
AUTHOR=Path('/root/autodl-fs/em_eqf_catalogs/earthquakenpp_20261106/author_snapshot')
SPECS={
 'ENPP_ComCat25':dict(file='ComCat/ComCat_catalog.csv',mc=2.5,start='1971-01-01',train='1981-01-01',val='1998-01-01',test='2007-01-01',end='2020-01-17'),
 'ENPP_SCEDC20':dict(file='SCEDC/SCEDC_catalog.csv',mc=2.,start='1981-01-01',train='1985-01-01',val='2005-01-01',test='2014-01-01',end='2020-01-01'),
 'ENPP_SCEDC25':dict(file='SCEDC/SCEDC_catalog.csv',mc=2.5,start='1981-01-01',train='1985-01-01',val='2005-01-01',test='2014-01-01',end='2020-01-01'),
 'ENPP_SCEDC30':dict(file='SCEDC/SCEDC_catalog.csv',mc=3.,start='1981-01-01',train='1985-01-01',val='2005-01-01',test='2014-01-01',end='2020-01-01'),
 'ENPP_QTMSanJacinto10':dict(file='QTM/SanJac_catalog.csv',mc=1.,start='2008-01-01',train='2009-01-01',val='2014-01-01',test='2016-01-01',end='2018-01-01'),
 'ENPP_QTMSaltonSea10':dict(file='QTM/SaltonSea_catalog.csv',mc=1.,start='2008-01-01',train='2009-01-01',val='2014-01-01',test='2016-01-01',end='2018-01-01'),
 'ENPP_White06':dict(file='WHITE/WHITE_catalog.csv',mc=.6,start='2008-01-01',train='2009-01-01',val='2014-01-01',test='2017-01-01',end='2021-01-01'),
 'ENPP_JapanDeprecated':dict(file='Japan_Deprecated/Japan_catalog.csv',mc=2.5,start='1990-01-01',train='1992-01-01',val='2007-01-01',test='2011-01-01',end='2020-01-01',deprecated=True),
}


def read_author_csv(path):
    f=pd.read_csv(path,dtype={'id':str,'index':str})
    if 'id' not in f:
        if 'index' not in f:raise ValueError('Author CSV has neither id nor index identity column')
        f['id']=f['index']
    return f


class PublishedEarthquakeNPP(Catalog):
    dataset_name=None
    schema='earthquakenpp-author-csv-f64-20261106-v1'
    timestamp_resolution_seconds=.000001

    def __init__(self,root_dir=None,data_dir=None,catalog_file=None,mag_completeness=None,
                 normalize=True,start_ts=None,end_ts=None,train_start_ts=None,val_start_ts=None,test_start_ts=None):
        s=SPECS[self.dataset_name];ds=Path(data_dir) if data_dir is not None else default_catalogs_dir/self.dataset_name
        self.catalog_file=(Path(catalog_file) if catalog_file is not None else AUTHOR/'Datasets'/s['file']).resolve()
        self.mc=s['mc'] if mag_completeness is None else float(mag_completeness)
        if self.mc<s['mc']:raise ValueError('Cannot lower threshold below the supplied benchmark definition')
        self.start=utc_naive(start_ts or s['start']);self.end=utc_naive(end_ts or s['end'])
        self.train_start=utc_naive(train_start_ts or s['train']);self.val_start=utc_naive(val_start_ts or s['val']);self.test_start=utc_naive(test_start_ts or s['test'])
        if not self.start<=self.train_start<self.val_start<self.test_start<self.end:raise ValueError('Invalid chronological protocol')
        self.normalize=bool(normalize)
        cfg=dict(dataset=self.dataset_name,schema=self.schema,source_sha256=file_sha256(self.catalog_file),
            author_commit=COMMIT,mag_completeness=self.mc,threshold_operator='>=',normalize=self.normalize,
            start_ts=self.start.isoformat(),train_start_ts=self.train_start.isoformat(),val_start_ts=self.val_start.isoformat(),
            test_start_ts=self.test_start.isoformat(),end_ts=self.end.isoformat(),
            loc_basis='author projected x/y km; train-only minmax when normalize=True',depth_available=False,
            timestamp_resolution_seconds=self.timestamp_resolution_seconds,time_processing='stable-bounded-f64-v1')
        metadata=dict(name=self.dataset_name,provider='EarthquakeNPP author snapshot',freq='1D',schema=self.schema,
            time_zone='UTC',source_sha256=cfg['source_sha256'],author_commit=COMMIT,
            mag_completeness=self.mc,mag_roundoff_error=0.0,start_ts=self.start,end_ts=self.end,
            train_start_ts=self.train_start,val_start_ts=self.val_start,test_start_ts=self.test_start,
            depth_available=False,loc_basis=cfg['loc_basis'],deprecated_compatibility_only=s.get('deprecated',False),
            quality_protocol='author-frozen-benchmark-integrity; not operational precision screening')
        cache=build_hashed_catalog_root(root_dir or ds/'catalogs/author_20261106',cfg,migrate_legacy=False)
        super().__init__(root_dir=cache,metadata=metadata)
        self.full_sequence=TppDataset.load_from_disk(self.root_dir/'full_sequence.pt')[0]
        self.norm_stats=torch.load(self.root_dir/'norm_stats.pt',weights_only=False)
        parts=train_val_test_split_sequence(self.full_sequence,self.start,self.train_start,self.val_start,self.test_start,half_open=True)
        self.train,self.val,self.test=[TppDataset([part]) for part in parts]

    @property
    def required_files(self):
        return ['full_sequence.pt','metadata.pt','norm_stats.pt','catalog.csv','catalog.parquet','processing_report.json']

    def load_source_frame(self):
        f=read_author_csv(self.catalog_file)
        required={'id','time','magnitude','longitude','latitude','x','y'}
        if required-set(f):raise ValueError(f'Missing author fields: {required-set(f)}')
        f['ts']=pd.to_datetime(f.time,utc=True,format='mixed',errors='raise').dt.tz_convert(None).dt.as_unit('ns')
        if f.id.isna().any() or f.id.duplicated().any():raise ValueError('Duplicate or missing author event IDs')
        if not np.isfinite(f[['magnitude','longitude','latitude','x','y']]).all().all():raise ValueError('Nonfinite author physical values')
        f=f[(f.ts>=self.start)&(f.ts<self.end)&(f.magnitude>=self.mc)].copy()
        return f.sort_values(['ts','id'],kind='stable').reset_index(drop=True)

    def generate_catalog(self):
        f=self.load_source_frame()
        raw=(f.ts.astype('int64').to_numpy()-self.start.value)/86400e9
        if not len(f) or raw[0]<=0:raise ValueError('Empty catalog or event on observation origin')
        adjusted=raw.copy()
        boundaries=[self.start,self.train_start,self.val_start,self.test_start,self.end]
        for lo,hi in zip(boundaries,boundaries[1:]):
            if lo==hi:continue
            mask=((f.ts>=lo)&(f.ts<hi)).to_numpy()
            if not mask.any():continue
            source=raw[mask]
            result=resolve_tied_times(source,self.timestamp_resolution_seconds/86400,(hi-self.start)/pd.Timedelta(days=1))
            lower=(lo-self.start)/pd.Timedelta(days=1)
            if result[0]<lower:
                count=int((source==lower).sum())
                upper=min(lower+self.timestamp_resolution_seconds/(2*86400),
                    result[count] if count<len(result) else (hi-self.start)/pd.Timedelta(days=1))
                result[:count]=lower+np.arange(count,dtype=np.float64)*(upper-lower)/count
            adjusted[mask]=result
        train=((f.ts>=self.train_start)&(f.ts<self.val_start)).to_numpy()
        if train.sum()<2:raise ValueError('Insufficient scored training events')
        stats={};locations=[]
        for field in ['x','y']:
            values=f[field].to_numpy(np.float64);low=float(values[train].min());high=float(values[train].max())
            stats[field]=dict(min=low,max=high)
            locations.append((values-low)/max(high-low,1e-8) if self.normalize else values)
        torch.save(stats,self.root_dir/'norm_stats.pt')
        inter=np.diff(adjusted,prepend=0.,append=(self.end-self.start)/pd.Timedelta(days=1))
        seq=Sequence(inter_times=torch.tensor(inter,dtype=torch.float64),mag=torch.tensor(f.magnitude.to_numpy(),dtype=torch.float32),
            loc=torch.tensor(np.stack(locations,axis=-1),dtype=torch.float32),event_index=torch.arange(len(f),dtype=torch.int64))
        TppDataset([seq]).save_to_disk(self.root_dir/'full_sequence.pt')
        f['event_id']=f.id;f['event_index']=np.arange(len(f));f['model_time_days']=adjusted
        f['time_adjustment_seconds']=(adjusted-raw)*86400
        f.to_parquet(self.root_dir/'catalog.parquet',index=False,compression='zstd');f.to_csv(self.root_dir/'catalog.csv',index=False)
        counts={name:int(((f.ts>=lo)&(f.ts<hi)).sum()) for name,lo,hi in
            [('auxiliary',self.start,self.train_start),('train',self.train_start,self.val_start),('val',self.val_start,self.test_start),('test',self.test_start,self.end)]}
        report=dict(records=len(f),scored_events=counts,depth_available=False,normalization_scope='training-scored-events-only',
            source_sha256=self.metadata['source_sha256'],author_commit=COMMIT,
            max_adjustment_seconds=float(f.time_adjustment_seconds.abs().max()),
            physical_time_preserved=True,source_projected_coordinates_preserved=True,
            feature_note='Projected x/y normalized using scored training events. No focal depth supplied or synthesized.')
        (self.root_dir/'processing_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


for _name in SPECS:
    _cls=type(_name,(PublishedEarthquakeNPP,),{'dataset_name':_name,'__module__':__name__})
    globals()[_name]=Catalog.register(_name+'-Standard')(_cls)
