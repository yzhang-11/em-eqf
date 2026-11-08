import numpy as np
import pandas as pd
import pytest

from scripts.data.download_regional_catalogs import sed_frame,quality_flags
from src.data import Catalog
import src.catalogs.regional_catalogs_20261105


def test_sed_preferred_origin_and_metre_conversion(tmp_path):
    xml='''<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2"><eventParameters><event publicID="e1"><type>earthquake</type><preferredOriginID>good</preferredOriginID><preferredMagnitudeID>m1</preferredMagnitudeID><origin publicID="bad"><depth><value>999999</value></depth></origin><origin publicID="good"><time><value>2020-01-01T00:00:01Z</value></time><latitude><value>47</value></latitude><longitude><value>8</value></longitude><depth><value>1250</value><uncertainty>500</uncertainty></depth><originUncertainty><horizontalUncertainty>750</horizontalUncertainty></originUncertainty><quality><usedStationCount>10</usedStationCount><azimuthalGap>90</azimuthalGap><standardError>0.1</standardError></quality><evaluationMode>manual</evaluationMode><evaluationStatus>confirmed</evaluationStatus></origin><magnitude publicID="m1"><mag><value>2.5</value></mag><type>MLhc</type></magnitude></event></eventParameters></q:quakeml>'''
    p=tmp_path/'sample.xml';p.write_text(xml)
    f=sed_frame(p)
    assert f.depth_km.iloc[0]==1.25
    assert f.depth_error_km.iloc[0]==.5
    assert f.horizontal_error_km.iloc[0]==.75
    keep,spatial,_=quality_flags(f,'SED_Switzerland')
    assert keep.iloc[0] and spatial.iloc[0]
    f.loc[0,'event_type']='quarry blast'
    assert not quality_flags(f,'SED_Switzerland')[0].iloc[0]


def test_missing_sed_preferred_origin_is_not_replaced(tmp_path):
    p=tmp_path/'x.xml'
    p.write_text('<quakeml xmlns="urn:test"><eventParameters><event publicID="e"><preferredOriginID>missing</preferredOriginID><origin publicID="alternative"><latitude><value>47</value></latitude></origin></event></eventParameters></quakeml>')
    f=sed_frame(p)
    assert np.isnan(f.latitude.iloc[0])
    assert f.evaluation_mode.iloc[0]==''


def test_alaska_missing_horizontal_quality_is_not_certified():
    f=pd.DataFrame(dict(evaluation_mode=['manual'],evaluation_status=['reviewed'],event_type=['earthquake'],
        magnitude=[3.2],station_count=[np.nan],azimuthal_gap=[np.nan],horizontal_error_km=[np.nan],
        rms=[.4],depth_error_km=[1.]))
    keep,spatial,_=quality_flags(f,'USGS_Alaska')
    assert keep.iloc[0] and not spatial.iloc[0]
    f.loc[0,'evaluation_status']='automatic'
    assert not quality_flags(f,'USGS_Alaska')[0].iloc[0]


@pytest.mark.parametrize('name',['USGS_Alaska','USGS_Hawaii','USGS_PuertoRico','SED_Switzerland'])
def test_added_registry_uses_existing_standard_adapter(name):
    cls=Catalog.by_name(name+'-Standard')
    assert cls.dataset_name==cls.provider==name
    assert cls.source_floor==cls.default_mc
    assert cls.default_start=='2014-01-01'
    assert cls.default_validation=='2019-01-01'
    assert cls.default_test=='2022-01-01'

