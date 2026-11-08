"""Meaningful source-policy and author-benchmark regression tests."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from scripts.data.normalize_research_20261106 import bit_event,screen_bit,screen_sw
from src.catalogs.earthquakenpp_20261106 import ENPP_ComCat25,SPECS,read_author_csv


def make_event(horizontal=True):
    h='<originUncertainty><horizontalUncertainty>800</horizontalUncertainty></originUncertainty>' if horizontal else ''
    return ET.fromstring(f'''<event publicID="e"><type>earthquake</type><preferredOriginID>p</preferredOriginID><preferredMagnitudeID>m</preferredMagnitudeID>
        <origin publicID="p"><time><value>2021-01-01T00:00:01Z</value></time><latitude><value>59.8</value></latitude><longitude><value>7.9</value></longitude><depth><value>6000</value><uncertainty>1200</uncertainty></depth>{h}</origin>
        <origin publicID="alternative"><originUncertainty><horizontalUncertainty>100</horizontalUncertainty></originUncertainty><quality><usedStationCount>100</usedStationCount><azimuthalGap>10</azimuthalGap><standardError>0.1</standardError></quality></origin>
        <magnitude publicID="m"><mag><value>0.6</value></mag><type>ML</type></magnitude></event>''')


def test_research_preferred_origin_does_not_borrow_alternative_quality():
    row=bit_event(make_event(False),Path('raw.xml'))
    assert np.isnan(row['horizontal_error_km']) and np.isnan(row['station_count'])
    assert not screen_bit(pd.DataFrame([row])).iloc[0]


def test_research_policy_preserves_unreported_review_and_station_fields():
    row=bit_event(make_event(True),Path('raw.xml'))
    assert row['depth_km']==6 and row['depth_error_km']==1.2 and row['horizontal_error_km']==.8
    assert row['evaluation_mode']==row['evaluation_status']=='' and not row['manual_review_verified']
    assert np.isnan(row['station_count']) and screen_bit(pd.DataFrame([row])).iloc[0]


def test_fixed_depth_source_flag_is_not_lost_from_precision_screen():
    row=dict(event_type='earthquake',mag_type='ML',magnitude=.5,station_count=10,azimuthal_gap=90,
        rms=.3,horizontal_error_km=1.,depth_error_km=1.,depth_indicator='F',location_indicator='',time_fixed=False)
    assert not screen_sw(pd.DataFrame([row])).iloc[0]
    row['depth_indicator']='';assert screen_sw(pd.DataFrame([row])).iloc[0]


@pytest.mark.parametrize('name',['id','index'])
def test_author_identity_columns_are_preserved(tmp_path,name):
    p=tmp_path/'author.csv';p.write_text(f'{name},time\n0007,2000-01-01\n')
    f=read_author_csv(p);assert f.id.iloc[0]=='0007' and f[name].iloc[0]=='0007'


def test_author_identity_is_not_invented_when_missing(tmp_path):
    p=tmp_path/'author.csv';p.write_text('time,magnitude\n2000-01-01,2.5\n')
    with pytest.raises(ValueError,match='neither id nor index'):read_author_csv(p)


def test_auxiliary_history_is_unscored_and_normalization_uses_train_only(tmp_path):
    times=['1975-01-01','1976-01-01','1985-01-01','1986-01-01',
        '1998-01-01T00:00:00','1998-01-01T00:00:00','2010-01-01','2011-01-01']
    f=pd.DataFrame(dict(id=[str(i) for i in range(8)],time=times,magnitude=[2.7]*8,
        latitude=[35.]*8,longitude=[-117.]*8,x=[1000,2000,1,2,3000,4000,5000,6000],y=[2000,3000,3,4,4000,5000,6000,7000]))
    path=tmp_path/'author.csv';f.to_csv(path,index=False)
    obj=ENPP_ComCat25(root_dir=tmp_path/'cache',catalog_file=path)
    assert obj.norm_stats=={'x':{'min':1.,'max':2.},'y':{'min':3.,'max':4.}}
    assert obj.metadata['depth_available'] is False and 'depth' not in obj.full_sequence
    expected={'train':{2,3},'val':{4,5},'test':{6,7}}
    for split in expected:
        seq=getattr(obj,split)[0]
        ids=set(seq.event_index[seq.arrival_times>=seq.t_nll_start].tolist())
        assert ids==expected[split]
    physical=pd.read_parquet(obj.root_dir/'catalog.parquet')
    assert physical.ts.iloc[4]==physical.ts.iloc[5]
    assert np.all(np.diff(obj.full_sequence.arrival_times.numpy())>0)


def test_japan_legacy_is_not_an_official_benchmark():
    assert len([k for k,v in SPECS.items() if not v.get('deprecated',False)])==7
    assert SPECS['ENPP_JapanDeprecated']['deprecated'] is True
