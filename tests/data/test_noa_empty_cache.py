"""Regression tests for a real false-empty response and incomplete repairs."""
from datetime import datetime
import json
from types import SimpleNamespace

import pytest

import scripts.data.download_more_catalogs as m


def response(status,body=b''):
    return SimpleNamespace(status_code=status,content=body,text=body.decode(),
        url='https://eida.gein.noa.gr/fdsnws/event/1/query',raise_for_status=lambda:None)


def empty_parent(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'STORE',tmp_path)
    monkeypatch.setattr(m.shared,'request',lambda *_:response(204))
    start,end=datetime(2019,12,1),datetime(2020,1,1)
    path,meta=m.fetch('NOA_Greece',start,end)[0]
    meta.pop('empty_verification')
    path.with_name(path.name+'.json').write_text(json.dumps(meta))
    return path,meta,start,end


def test_noa_legacy_empty_is_rejected_and_verified_empty_is_reused(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'STORE',tmp_path)
    calls=[]
    def get(_,params):
        calls.append(params['format']);return response(204)
    monkeypatch.setattr(m.shared,'request',get)
    start,end=datetime(2019,12,1),datetime(2020,1,1)
    path,meta=m.fetch('NOA_Greece',start,end)[0]
    assert calls==['xml','text']
    assert meta['empty_verification']['format']=='text'
    assert m.fetch('NOA_Greece',start,end)[0][0]==path
    assert calls==['xml','text']
    meta.pop('empty_verification');path.with_name(path.name+'.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Unverified empty NOA cache'):
        m.fetch('NOA_Greece',start,end)
    assert calls==['xml','text']


def test_noa_nonempty_text_contradicts_xml_204_and_is_not_cached(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'STORE',tmp_path)
    def get(_,params):
        return response(204) if params['format']=='xml' else response(200,b'#EventID|Time\ne|2019-12-01T01:02:03Z\n')
    monkeypatch.setattr(m.shared,'request',get)
    with pytest.raises(ValueError,match='contradicts nonempty text'):
        m.fetch('NOA_Greece',datetime(2019,12,1),datetime(2019,12,2))
    assert not list(tmp_path.rglob('*.xml'))


@pytest.mark.parametrize('complete',[True,False])
def test_repaired_month_requires_complete_contiguous_coverage(monkeypatch,tmp_path,complete):
    path,parent,start,end=empty_parent(monkeypatch,tmp_path)
    monkeypatch.setattr(m.shared,'request',lambda *_:response(200,b'<quakeml><eventParameters><event publicID="e"/></eventParameters></quakeml>'))
    mid=datetime(2019,12,16)
    parts=m.fetch('NOA_Greece',start,mid)
    if complete:parts+=m.fetch('NOA_Greece',mid,end)
    repair=dict(parent_sha256=parent['sha256'],replacements=[meta for _,meta in parts])
    path.with_name(path.name+'.replacements.json').write_text(json.dumps(repair))
    if complete:
        got=m.fetch('NOA_Greece',start,end)
        assert [p for p,_ in got]==[p for p,_ in parts]
        assert m.shared.sha(path)==parent['sha256']
    else:
        with pytest.raises(AssertionError,match='do not cover'):
            m.fetch('NOA_Greece',start,end)
