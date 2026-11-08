import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.catalogs.external_catalog import JMAKyushu, file_sha256


def snapshot(path, **updates):
    times=['2012-01-02','2013-01-02','2014-01-01','2014-01-01',
           '2015-03-01','2016-01-01','2016-01-01','2017-01-02','2018-01-01']
    f=pd.DataFrame(dict(time=pd.to_datetime(times,utc=True).as_unit('us'),
        magnitude=[3.,3.2,4.,4.1,4.2,5.,5.1,5.2,5.3],
        latitude=[30.,31.,40.,41.,42.,50.,51.,52.,53.],
        longitude=[130.,131.,140.,141.,142.,150.,151.,152.,153.],
        depth_km=[1.,2.,10.,11.,12.,20.,21.,22.,23.],event_id=[str(i) for i in range(9)],
        provider=['JMA']*9,absolute_time_ready=[True]*9,quality_screen_pass=[True]*9,
        mag_type=['V']*9))
    for k,v in updates.items():
        f[k]=v
    f.to_parquet(path,index=False)
    return f


def make(tmp_path, source, **kwargs):
    cfg=dict(root_dir=tmp_path/'catalogs',catalog_file=source,start_ts='2012-01-01',
             end_ts='2018-01-01',val_start_ts='2014-01-01',test_start_ts='2016-01-01')
    cfg.update(kwargs)
    return JMAKyushu(**cfg)


def targets(seq):
    return set(seq.event_index[seq.arrival_times>=seq.t_nll_start].tolist())


def test_training_only_normalization_and_statistics(tmp_path):
    p=tmp_path/'source.parquet';snapshot(p)
    obj=make(tmp_path,p)
    assert obj.norm_stats['latitude']=={'min':30.,'max':31.}
    assert obj.norm_stats['depth']=={'min':1.,'max':2.}
    assert obj.training_statistics['events']==2
    assert obj.training_statistics['magnitude_mean']==pytest.approx(3.1)
    assert obj.full_sequence.loc[-1,0]>1  # Future extremes are not used or clipped.
    frame=pd.read_parquet(obj.root_dir/'catalog.parquet')
    assert (frame.ts.dt.year<2018).all()


def test_boundary_ties_preserved_and_scored_splits_disjoint(tmp_path):
    p=tmp_path/'source.parquet';snapshot(p)
    obj=make(tmp_path,p)
    assert len(obj.full_sequence)==8
    assert torch.all(obj.full_sequence.inter_times[:-1]>0)
    report=json.loads((obj.root_dir/'processing_report.json').read_text())
    assert report['tied_rows_retained']==2
    assert report['max_adjustment_seconds']<=.0050001
    a,b,c=[targets(x[0]) for x in [obj.train,obj.val,obj.test]]
    assert [len(a),len(b),len(c)]==[2,3,3]
    assert not a&b and not b&c and not a&c
    assert a|b|c==set(range(8))
    assert obj.full_sequence.inter_times.dtype==torch.float64
    assert obj.full_sequence.arrival_times[0].item()==pytest.approx(1.)


def test_cache_identity_includes_source_normalization_and_split(tmp_path):
    p=tmp_path/'source.parquet';snapshot(p)
    first=make(tmp_path,p)
    artifact_hash=file_sha256(first.root_dir/'full_sequence.pt')
    same=make(tmp_path,p)
    assert same.root_dir==first.root_dir
    assert file_sha256(same.root_dir/'full_sequence.pt')==artifact_hash
    assert make(tmp_path,p,normalize=False).root_dir!=first.root_dir
    assert make(tmp_path,p,val_start_ts='2013-07-01').root_dir!=first.root_dir
    f=pd.read_parquet(p);f.loc[0,'magnitude']=3.1;f.to_parquet(p,index=False)
    assert make(tmp_path,p).root_dir!=first.root_dir


def test_unconfirmed_time_blocks_training(tmp_path):
    p=tmp_path/'source.parquet';snapshot(p,absolute_time_ready=False)
    with pytest.raises(ValueError,match='unconfirmed'):
        make(tmp_path,p)


def test_duplicate_ids_fail_instead_of_deleting_events(tmp_path):
    p=tmp_path/'source.parquet';snapshot(p,event_id=['same']*9)
    with pytest.raises(ValueError,match='Duplicate event IDs'):
        make(tmp_path,p)


def test_invalid_cutoff_and_reversed_split_fail(tmp_path):
    p=tmp_path/'source.parquet';snapshot(p)
    with pytest.raises(ValueError,match='source screening floor'):
        make(tmp_path,p,mag_completeness=1.5)
    with pytest.raises(ValueError,match='Expected start'):
        make(tmp_path,p,test_start_ts='2013-01-01')

