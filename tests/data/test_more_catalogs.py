import numpy as np
import pandas as pd
import pytest

from scripts.data.download_more_catalogs import parse_quakeml,quality_flags,periods,SOURCES
from src.data import Catalog
import src.catalogs.more_catalogs_20261105


def test_ellipse_maximum_is_used_in_km_and_unknown_type_is_rejected(tmp_path):
    p=tmp_path/'x.xml'
    p.write_text('''<quakeml xmlns="urn:test"><eventParameters><event publicID="e"><type>earthquake</type><preferredOriginID>o</preferredOriginID><preferredMagnitudeID>m</preferredMagnitudeID><origin publicID="o"><time><value>2019-01-01T00:00:01Z</value></time><latitude><value>64</value></latitude><longitude><value>-21.4</value></longitude><depth><value>5000</value><uncertainty>1000</uncertainty></depth><originUncertainty><maxHorizontalUncertainty>1500</maxHorizontalUncertainty></originUncertainty><quality><usedStationCount>20</usedStationCount><azimuthalGap>100</azimuthalGap><standardError>0.1</standardError></quality><evaluationMode>manual</evaluationMode><evaluationStatus>confirmed</evaluationStatus></origin><magnitude publicID="m"><mag><value>1.5</value></mag><type>ML</type></magnitude></event></eventParameters></quakeml>''')
    f=parse_quakeml(p)
    assert f.horizontal_error_km.iloc[0]==1.5
    assert 'maxHorizontal' in f.horizontal_error_basis.iloc[0]
    assert quality_flags(f,'ISOR_Hengill')[0].iloc[0]
    f.loc[0,'event_type']=''
    assert not quality_flags(f,'ISOR_Hengill')[0].iloc[0]
    f.loc[0,'event_type']='earthquake';f.loc[0,'horizontal_error_km']=np.nan
    assert not quality_flags(f,'ISOR_Hengill')[0].iloc[0]


def test_iceland_periods_cover_only_actual_archive_and_are_contiguous():
    parts=periods(SOURCES['ISOR_Hengill'])
    assert len(parts)==26
    assert str(parts[0][0])=='2018-12-01 00:00:00'
    assert str(parts[-1][1])=='2021-02-01 00:00:00'
    assert all(a[1]==b[0] for a,b in zip(parts,parts[1:]))
    cls=Catalog.by_name('ISOR_Hengill-Standard')
    assert cls.default_validation=='2020-01-01'
    assert cls.default_test=='2020-07-01'


def test_greece_uses_monthly_time_bounded_requests():
    parts=periods(SOURCES['NOA_Greece'])
    assert len(parts)==144
    assert all((b-a).days<=31 for a,b in parts)
    assert Catalog.by_name('NOA_Greece-Standard').default_start=='2018-01-01'


@pytest.mark.parametrize('name',['USGS_PNW','USGS_Utah','USGS_NewMadrid','NOA_Greece','ISOR_Hengill'])
def test_registry(name):
    cls=Catalog.by_name(name+'-Standard')
    assert cls.provider==name
    assert cls.default_mc==cls.source_floor
