"""Coverage, immutable native provenance and interrupted-conversion regression tests."""
from datetime import datetime,timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import scripts.data.resume_noa_catalog as r


@pytest.fixture
def isolated(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'STORE',tmp_path/'data')
    monkeypatch.setattr(r,'PLANS',tmp_path/'plans')
    monkeypatch.setattr(r,'OUT',tmp_path/'reports')
    return tmp_path


def part(tmp_path,a,b,native=False):
    path=tmp_path/f'{a:%Y%m%dT%H%M%S}.xml';path.write_bytes(b'<q><event publicID="e"/></q>')
    meta=dict(path=str(path),sha256=r.m.shared.sha(path),bytes=path.stat().st_size,
        start=a.isoformat(),end_exclusive=b.isoformat(),params=r.params_for(a,b),rows=1)
    if native:
        raw=tmp_path/'native.xml';raw.write_bytes(b'<seiscomp/>')
        meta['native_source']=dict(path=str(raw),sha256=r.m.shared.sha(raw),bytes=raw.stat().st_size,
            params=r.params_for(a,b,'sc3ml'))
    return meta


@pytest.mark.parametrize('offset',[timedelta(seconds=1),timedelta(seconds=-1)])
def test_gap_or_overlap_cannot_be_marked_complete(isolated,offset):
    a,mid,b=datetime(2020,1,1),datetime(2020,1,10),datetime(2020,2,1)
    entries=[part(isolated,a,mid),part(isolated,mid+offset,b)]
    with pytest.raises(ValueError,match='gap or overlap'):r.verify_parts(entries,a,b)


def test_tampered_native_payload_fails_even_if_quakeml_is_unchanged(isolated):
    a,b=datetime(2020,1,1),datetime(2020,2,1)
    meta=part(isolated,a,b,native=True)
    Path(meta['native_source']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Native source changed'):r.verify_parts([meta],a,b)


def test_complete_month_is_reused_without_http(isolated,monkeypatch):
    a,b=datetime(2020,1,1),datetime(2020,2,1)
    meta=part(isolated,a,b)
    r.publish_plan(a,b,[meta])
    monkeypatch.setattr(r,'http_get',lambda *a,**kw:pytest.fail('Unexpected network request'))
    assert r.fetch_window(a,b)==[meta]


def test_interrupted_conversion_reuses_native_bytes_without_http(isolated,monkeypatch):
    a,b=datetime(2020,1,1),datetime(2020,2,1)
    stem=f'{a:%Y%m%dT%H%M%S}_{b:%Y%m%dT%H%M%S}_{r.key_for(a,b)}'
    raw=r.STORE/'raw/native_sc3ml'/f'{stem}.sc3ml.xml';raw.parent.mkdir(parents=True)
    raw.write_bytes(b'<seiscomp><EventParameters><event publicID="e"/></EventParameters></seiscomp>')
    meta=dict(path=str(raw),sha256=r.m.shared.sha(raw),bytes=raw.stat().st_size,
        status=200,url='https://eida.gein.noa.gr/fdsnws/event/1/query',params=r.params_for(a,b,'sc3ml'))
    raw.with_name(raw.name+'.json').write_text(json.dumps(meta))
    class Converted:
        def __len__(self):return 1
        def write(self,path,format):Path(path).write_bytes(b'<quakeml><eventParameters><event publicID="e"/></eventParameters></quakeml>')
    monkeypatch.setattr(r.obspy,'read_events',lambda *a,**kw:Converted())
    monkeypatch.setattr(r,'http_get',lambda *a,**kw:pytest.fail('Unexpected re-download'))
    got=r.fetch_window(a,b)
    assert got[0]['rows']==1 and got[0]['native_source']['sha256']==meta['sha256']


def test_native_empty_with_nonempty_text_does_not_publish_coverage(isolated,monkeypatch):
    def get(url,params,**kwargs):
        text=params['format']=='text';body=b'#EventID|Time\ne|2020-01-01T00:00:01Z\n' if text else b''
        return SimpleNamespace(status_code=200 if text else 204,content=body,text=body.decode(),
            url=url,headers={},raise_for_status=lambda:None)
    monkeypatch.setattr(r,'http_get',get)
    a,b=datetime(2020,1,1),datetime(2020,1,2)
    with pytest.raises(ValueError,match='contradicts text'):r.fetch_window(a,b)
    assert not r.plan_path(a,b).exists()


def test_wire_request_with_different_magnitude_filter_is_rejected(isolated):
    a,b=datetime(2020,1,1),datetime(2020,2,1)
    meta=part(isolated,a,b,native=True)
    meta['native_source']['params']['minmagnitude']=3.0
    with pytest.raises(ValueError,match='Native request differs: minmagnitude'):
        r.verify_parts([meta],a,b)


@pytest.mark.parametrize('error',[requests.ConnectTimeout('connect'),requests.exceptions.SSLError('TLS')])
def test_connection_fault_does_not_split_time_window(isolated,monkeypatch,error):
    def fail(*a,**kw):raise error
    monkeypatch.setattr(r,'http_get',fail)
    a,b=datetime(2020,1,1),datetime(2020,2,1)
    with pytest.raises(type(error)):r.fetch_window(a,b)
    assert not r.plan_path(a,b).exists()


def test_http_200_with_embedded_413_splits_and_preserves_failure(isolated,monkeypatch):
    body=b'<seiscomp>partial response\nError 413: Request Entity Too Large\nmaximum number of objects (5000)'
    monkeypatch.setattr(r,'http_get',lambda url,params:SimpleNamespace(status_code=200,content=body,
        text=body.decode(),url=url,headers={},raise_for_status=lambda:None))
    class SplitRequested(Exception):pass
    def split(a,b,reason):raise SplitRequested(reason)
    monkeypatch.setattr(r,'split_window',split)
    a,b=datetime(2020,1,1),datetime(2020,2,1)
    with pytest.raises(SplitRequested,match='HTTP 200 contains service error 413'):
        r.fetch_window(a,b)
    records=list((r.STORE/'raw/response_failures').glob('*.body.json'))
    assert len(records)==1 and json.loads(records[0].read_text())['service_error']==413
    assert not r.plan_path(a,b).exists()


def test_second_downloader_cannot_take_catalog_lock(isolated):
    owner=r.acquire_download_lock()
    try:
        with pytest.raises(RuntimeError,match='Another NOA downloader'):r.acquire_download_lock()
    finally:owner.close()
    released=r.acquire_download_lock();released.close()
