"""Further public catalog snapshots with per-source time windows.

The validated earlier parser/curator is loaded into an isolated module instance
with this batch's configuration. No previous source file is modified.
"""
import argparse
import concurrent.futures as cf
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
STORE=Path('/root/autodl-fs/em_eqf_catalogs/more_20261105')
OUT=ROOT/'experiments/revision/catalog_more_20261105'
SOURCES={
 'USGS_PNW':dict(kind='usgs',url='https://earthquake.usgs.gov/fdsnws/event/1/query',contributor='uw',
   box=(42.,50.,-127.,-116.),mc=2.,start='2014-01-01',end='2026-01-01',
   reference='https://earthquake.usgs.gov/data/comcat/catalog/uw/'),
 'USGS_Utah':dict(kind='usgs',url='https://earthquake.usgs.gov/fdsnws/event/1/query',contributor='uu',
   box=(36.,45.,-115.,-108.),mc=2.,start='2014-01-01',end='2026-01-01',
   reference='https://earthquake.usgs.gov/data/comcat/catalog/uu/'),
 'NOA_Greece':dict(kind='qml',url='https://eida.gein.noa.gr/fdsnws/event/1/query',
   box=(33.,42.,19.,30.),mc=2.5,fetch_minmagnitude=2.499,start='2014-01-01',end='2026-01-01',monthly=True,
   reference='https://eida.gein.noa.gr/fdsnws/event/1/'),
 'ISOR_Hengill':dict(kind='qml',url='http://coseismiq.ethz.ch:8080/fdsnws/event/1/query',contributor='ISOR_manual',
   box=(63.9,64.2,-21.7,-20.9),mc=1.,start='2018-12-01',end='2021-02-01',monthly=True,
   reference='https://www.coseismiq.ethz.ch/en/dissemination/catalogs/'),
}

spec=importlib.util.spec_from_file_location('_more_catalog_shared',ROOT/'scripts/data/download_regional_catalogs.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
shared.ROOT=ROOT;shared.OUT=OUT;shared.STORE=STORE;shared.SOURCES=SOURCES


def parse_quakeml(path):
    f=shared_original_parser(path)
    # Some ISOR solutions only report the horizontal ellipse semiaxes. Use the
    # reported maximum as a conservative size screen, never infer from degrees.
    extra={}
    for e in ET.parse(path).findall('.//{*}event'):
        oid=shared.text(e,'{*}preferredOriginID')
        o=next((v for v in e.findall('{*}origin') if v.get('publicID')==oid),None)
        horizontal=shared.number(o,'{*}originUncertainty/{*}horizontalUncertainty',1000)
        maximum=shared.number(o,'{*}originUncertainty/{*}maxHorizontalUncertainty',1000)
        if np.isfinite(horizontal):value,basis=horizontal,'horizontalUncertainty_m_to_km'
        elif np.isfinite(maximum):value,basis=maximum,'reported_maxHorizontalUncertainty_m_to_km'
        else:value,basis=np.nan,'missing'
        extra[e.get('publicID')]=(value,basis,maximum)
    if len(f):
        f['horizontal_error_km']=[extra[v][0] for v in f.event_id]
        f['horizontal_error_basis']=[extra[v][1] for v in f.event_id]
        f['max_horizontal_uncertainty_km']=[extra[v][2] for v in f.event_id]
    return f


shared_original_parser=shared.sed_frame
shared.sed_frame=parse_quakeml


def quality_flags(f,provider):
    reviewed=f.evaluation_mode.eq('manual')&f.evaluation_status.isin(['reviewed','confirmed','final'])
    spatial=((f.station_count>=8)&(f.azimuthal_gap<=180)&(f.rms<=1)&
             (f.horizontal_error_km<=2)&(f.depth_error_km<=5)).fillna(False)
    keep=reviewed&f.event_type.eq('earthquake')&(f.magnitude>=SOURCES[provider]['mc'])&spatial
    policy=f'preferred origin manual reviewed/confirmed/final; explicit earthquake; M>={SOURCES[provider]["mc"]}; stations>=8; gap<=180deg; RMS<=1s; reported horizontal uncertainty (or maximum ellipse size)<=2km; depth uncertainty<=5km. Missing quality fields fail.'
    return keep.fillna(False),spatial,policy


shared.quality_flags=quality_flags


def verified_replacements(path,meta,start,end):
    """Follow an explicit audited repair without overwriting the old response."""
    index=path.with_name(path.name+'.replacements.json')
    if not index.exists():
        raise ValueError(f'Unverified empty NOA cache: {path}; revalidate or repair before resuming')
    repair=json.loads(index.read_text())
    assert repair['parent_sha256']==meta['sha256']
    parts=sorted(repair['replacements'],key=lambda v:v['start'])
    cursor=start.isoformat();result=[]
    for part in parts:
        assert part['start']==cursor,'Replacement intervals are not contiguous'
        assert part['params']['starttime']==part['start']
        stop=datetime.fromisoformat(part['end_exclusive'])
        assert stop>datetime.fromisoformat(part['start'])
        assert part['params']['endtime']==(stop-timedelta(microseconds=1)).isoformat()
        raw=Path(part['path']);assert shared.sha(raw)==part['sha256']
        assert raw.stat().st_size==part['bytes']
        assert part['bytes'] and len(ET.parse(raw).findall('.//{*}event'))>0
        for key,value in meta['params'].items():
            if key not in ('starttime','endtime'):assert part['params'][key]==value
        cursor=part['end_exclusive'];result.append((raw,part))
    assert cursor==end.isoformat(),'Replacement intervals do not cover the parent query'
    return result


def fetch(provider,start,end,depth=0):
    cfg=SOURCES[provider];la,lb,lo,lp=cfg['box'];usgs=cfg['kind']=='usgs'
    params=dict(starttime=start.isoformat(),endtime=(end-timedelta(microseconds=1)).isoformat(),
        minlatitude=la,maxlatitude=lb,minlongitude=lo,maxlongitude=lp,
        format='csv' if usgs else 'xml',orderby='time-asc')
    if 'contributor' in cfg:params['contributor']=cfg['contributor']
    if usgs:params.update(eventtype='earthquake',reviewstatus='reviewed')
    if 'fetch_minmagnitude' in cfg:params['minmagnitude']=cfg['fetch_minmagnitude']
    key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:16]
    path=STORE/provider/'raw'/f'{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_{key}.{"csv" if usgs else "xml"}'
    meta_path=path.with_name(path.name+'.json')
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text());assert shared.sha(path)==meta['sha256']
        if provider=='NOA_Greece' and not meta['bytes'] and not meta.get('empty_verification'):
            return verified_replacements(path,meta,start,end)
    else:
        response=shared.request(cfg['url'],params)
        if response.status_code in (400,413) and re.search(r'limit|maximum|exceed|too large|too many',response.text,re.I):
            shared.log('SPLIT_REQUIRED',provider,start.isoformat(),end.isoformat(),response.text[:160].replace('\n',' '))
            if depth>=18 or (end-start).total_seconds()<=1:raise ValueError('Unresolved query limit')
            mid=(start+(end-start)/2).replace(microsecond=0)
            return fetch(provider,start,mid,depth+1)+fetch(provider,mid,end,depth+1)
        if response.status_code!=204:response.raise_for_status()
        content=b'' if response.status_code==204 else response.content
        if content.lstrip().lower().startswith((b'<html',b'<!doctype html')):raise ValueError('HTML catalog response')
        empty_verification=None
        if provider=='NOA_Greece' and (not content or not ET.fromstring(content).findall('.//{*}event')):
            # A historic 204 was contradicted by a fresh nonempty day query.
            # Never accept an empty XML response without an independent format.
            confirm=shared.request(cfg['url'],dict(params,format='text'))
            if confirm.status_code!=204:confirm.raise_for_status()
            records=[v for v in confirm.text.splitlines()
                     if v.strip() and not v.lstrip().startswith('#')
                     and not v.lower().startswith('eventid|')]
            if records:raise ValueError('NOA empty XML response contradicts nonempty text response')
            empty_verification=dict(format='text',status=confirm.status_code,url=confirm.url,
                sha256=hashlib.sha256(confirm.content).hexdigest(),
                checked_utc=datetime.now(timezone.utc).isoformat())
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.part');tmp.write_bytes(content);tmp.replace(path)
        meta=dict(path=str(path),url=response.url,params=params,bytes=len(content),sha256=shared.sha(path),
            status=response.status_code,retrieved_utc=datetime.now(timezone.utc).isoformat(),
            start=start.isoformat(),end_exclusive=end.isoformat())
        if empty_verification is not None:meta['empty_verification']=empty_verification
        shared.dump(meta_path,meta)
    n=(len(pd.read_csv(path)) if usgs else len(ET.parse(path).findall('.//{*}event'))) if meta['bytes'] else 0
    if provider=='NOA_Greece' and n==0 and not meta.get('empty_verification'):
        return verified_replacements(path,meta,start,end)
    if n>=19000:
        if depth>=18:raise ValueError('Unresolved possible response cap')
        mid=(start+(end-start)/2).replace(microsecond=0)
        return fetch(provider,start,mid,depth+1)+fetch(provider,mid,end,depth+1)
    meta['rows']=n
    shared.log('DOWNLOADED',provider,start.isoformat(),end.isoformat(),n)
    return [(path,meta)]


def periods(cfg):
    start=pd.Timestamp(cfg['start']);end=pd.Timestamp(cfg['end'])
    boundaries=pd.date_range(start,end,freq='MS' if cfg.get('monthly') else 'YS')
    points=sorted(set([start,end,*boundaries]))
    return [(a.to_pydatetime(),b.to_pydatetime()) for a,b in zip(points,points[1:])]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='+',choices=list(SOURCES),help='Resume selected sources; preserve other completed audits.')
    parser.add_argument('--workers',type=int,choices=range(1,5),default=2)
    args=parser.parse_args()
    (OUT/'shared/reports').mkdir(parents=True,exist_ok=True)
    previous=OUT/'reports/download_summary.json'
    previous=json.loads(previous.read_text()) if previous.exists() else {}
    results=previous.get('sources',{});errors=previous.get('errors',{})
    staged=previous.get('staged_downloads',{})
    for name in args.only or SOURCES:
        cfg=SOURCES[name]
        try:
            shared.log('START',name);paths=[]
            with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
                jobs=[pool.submit(fetch,name,a,b) for a,b in periods(cfg)]
                try:
                    for job in cf.as_completed(jobs):paths.extend(job.result())
                except BaseException:
                    # Stop pending requests if the service fails; do not spend
                    # hours waiting for every queued month to hit its timeout.
                    for job in jobs:job.cancel()
                    raise
            f=shared.load(name,sorted(paths,key=lambda v:str(v[0])))
            results[name]=shared.curate(name,f,paths)
            errors.pop(name,None);staged.pop(name,None)
        except Exception:
            errors[name]=traceback.format_exc();shared.log('FAILED',name,errors[name])
        shared.dump(OUT/'reports/download_progress.json',dict(completed=list(results),errors=errors))
    shared.dump(OUT/'reports/download_summary.json',dict(sources=results,errors=errors,
        staged_downloads=staged,
        shared_processor_sha256=shared.sha(ROOT/'scripts/data/download_regional_catalogs.py')))
    if errors:raise SystemExit(1)
    shared.log('ALL_DOWNLOADED')


if __name__=='__main__':main()
