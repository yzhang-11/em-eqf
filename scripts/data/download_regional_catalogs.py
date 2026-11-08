"""Download and audit four additional reviewed regional earthquake catalogs.

Raw responses are immutable cached snapshots with request metadata and checksums.
Unknown event types/review states never become earthquakes by default.
"""
import concurrent.futures as cf
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[2]
STORE=Path('/root/autodl-fs/em_eqf_catalogs/additions_20261105')
OUT=ROOT/'experiments/revision/catalog_additions_20261105'
SOURCES={
    'USGS_Hawaii':dict(kind='usgs',contributor='hv',box=(18,23,-161,-154),start=1985,mc=2.5,
        reference='https://earthquake.usgs.gov/data/comcat/catalog/hv/'),
    'USGS_PuertoRico':dict(kind='usgs',contributor='pr',box=(16,21,-70,-62),start=1985,mc=2.5,
        reference='https://earthquake.usgs.gov/data/comcat/catalog/pr/'),
    'USGS_Alaska':dict(kind='usgs',contributor='ak',box=(50,73,-190,-129),start=1985,mc=3.0,
        reference='https://earthquake.usgs.gov/data/comcat/catalog/ak/'),
    'SED_Switzerland':dict(kind='sed',box=(45.4,48.3,5.6,11.1),start=2009,mc=2.0,
        reference='https://www.seismo.ethz.ch/en/knowledge/earthquake-data-and-analysis-tools/fdsn-web-services/'),
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def dump(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.part')
    temp.write_text(json.dumps(x,indent=2,default=str,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)


def log(*args):
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'),*args,flush=True)


def request(url,params):
    for attempt in range(3):
        try:
            r=requests.get(url,params=params,timeout=(15,90))
            if r.status_code in (429,500,502,503,504) and attempt<2:
                time.sleep(2**(attempt+1));continue
            return r
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(2**(attempt+1))


def fetch(provider,start,end,depth=0):
    cfg=SOURCES[provider];la,lb,lo,lp=cfg['box']
    params=dict(starttime=start.isoformat(),endtime=(end-timedelta(microseconds=1)).isoformat(),
                minlatitude=la,maxlatitude=lb,minlongitude=lo,maxlongitude=lp)
    usgs=cfg['kind']=='usgs'
    url=('https://earthquake.usgs.gov/fdsnws/event/1/query' if usgs else
         'https://eida.ethz.ch/fdsnws/event/1/query')
    params.update(dict(format='csv' if usgs else 'xml',orderby='time-asc'))
    if usgs:params.update(contributor=cfg['contributor'],eventtype='earthquake',reviewstatus='reviewed')
    key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:16]
    path=STORE/provider/'raw'/f'{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_{key}.{"csv" if usgs else "xml"}'
    meta_path=path.with_name(path.name+'.json')
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text());assert sha(path)==meta['sha256']
    else:
        r=request(url,params)
        saturated=(r.status_code in (400,413) and re.search(r'limit|maximum|exceed|too large|too many',r.text,re.I))
        if saturated:
            if depth>=18 or (end-start).total_seconds()<=1:raise ValueError(f'Unsplittable saturated query: {r.url}')
            middle=(start+(end-start)/2).replace(microsecond=0)
            return fetch(provider,start,middle,depth+1)+fetch(provider,middle,end,depth+1)
        if r.status_code!=204:r.raise_for_status()
        payload=b'' if r.status_code==204 else r.content
        if payload and payload.lstrip().lower().startswith((b'<html',b'<!doctype html')):
            raise ValueError('HTML instead of earthquake data')
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.part');temp.write_bytes(payload);temp.replace(path)
        meta=dict(path=str(path),url=r.url,params=params,bytes=len(payload),sha256=sha(path),
                  status=r.status_code,retrieved_utc=datetime.now(timezone.utc).isoformat(),
                  start=start.isoformat(),end_exclusive=end.isoformat())
        dump(meta_path,meta)
    if meta['bytes']:
        n=len(pd.read_csv(path)) if usgs else len(ET.parse(path).findall('.//{*}event'))
    else:n=0
    if usgs and n>=20000:
        if depth>=18:raise ValueError('Unresolved event cap')
        middle=(start+(end-start)/2).replace(microsecond=0)
        return fetch(provider,start,middle,depth+1)+fetch(provider,middle,end,depth+1)
    meta['rows']=n
    log('DOWNLOADED',provider,start.isoformat(),end.isoformat(),n)
    return [(path,meta)]


def text(node,path,default=''):
    return node.findtext(path,default=default) if node is not None else default


def number(node,path,scale=1):
    raw=text(node,path)
    return float(raw)/scale if raw.strip() else np.nan


def sed_frame(path):
    rows=[]
    for e in ET.parse(path).findall('.//{*}event'):
        oid=text(e,'{*}preferredOriginID');mid=text(e,'{*}preferredMagnitudeID')
        # A missing preferred object is not replaced with the first alternative.
        origin=next((o for o in e.findall('{*}origin') if o.get('publicID')==oid),None)
        mag=next((o for o in e.findall('{*}magnitude') if o.get('publicID')==mid),None)
        rows.append(dict(event_id=e.get('publicID'),time=text(origin,'{*}time/{*}value'),
            latitude=number(origin,'{*}latitude/{*}value'),longitude=number(origin,'{*}longitude/{*}value'),
            depth_km=number(origin,'{*}depth/{*}value',1000),magnitude=number(mag,'{*}mag/{*}value'),
            mag_type=text(mag,'{*}type'),event_type=text(e,'{*}type'),
            evaluation_mode=text(origin,'{*}evaluationMode'),evaluation_status=text(origin,'{*}evaluationStatus'),
            station_count=number(origin,'{*}quality/{*}usedStationCount'),
            phase_count=number(origin,'{*}quality/{*}usedPhaseCount'),
            azimuthal_gap=number(origin,'{*}quality/{*}azimuthalGap'),rms=number(origin,'{*}quality/{*}standardError'),
            horizontal_error_km=number(origin,'{*}originUncertainty/{*}horizontalUncertainty',1000),
            depth_error_km=number(origin,'{*}depth/{*}uncertainty',1000),
            source_agency=text(origin,'{*}creationInfo/{*}agencyID'),raw_file=str(path)))
    return pd.DataFrame(rows)


def load(provider,paths):
    frames=[]
    for p,m in paths:
        if not m['bytes']:continue
        if SOURCES[provider]['kind']=='usgs':
            f=pd.read_csv(p).rename(columns={'id':'event_id','mag':'magnitude','magType':'mag_type',
                'depth':'depth_km','type':'event_type','status':'evaluation_status','nst':'station_count',
                'gap':'azimuthal_gap','horizontalError':'horizontal_error_km','depthError':'depth_error_km',
                'net':'source_agency'})
            f['evaluation_mode']=np.where(f.evaluation_status.eq('reviewed'),'manual','unconfirmed')
            f['raw_file']=str(p)
        else:f=sed_frame(p)
        if not len(f):continue
        f['time']=pd.to_datetime(f.time,utc=True,format='mixed',errors='coerce').dt.as_unit('ns')
        inside=(f.time>=pd.Timestamp(m['start'],tz='UTC'))&(f.time<pd.Timestamp(m['end_exclusive'],tz='UTC'))
        f=f[f.time.isna()|inside]  # Count unparseable/missing origin times in the rejection audit.
        frames.append(f)
    if not frames:raise ValueError(f'No downloaded events for {provider}')
    return pd.concat(frames,ignore_index=True)


def quality_flags(f,provider):
    mc=SOURCES[provider]['mc']
    reviewed=f.evaluation_mode.eq('manual') & f.evaluation_status.isin(['reviewed','confirmed','final'])
    common=reviewed & f.event_type.eq('earthquake') & (f.magnitude>=mc)
    spatial=((f.station_count>=8)&(f.azimuthal_gap<=180)&(f.rms<=1)&
             (f.horizontal_error_km<=2)&(f.depth_error_km<=5)).fillna(False)
    if provider=='USGS_Alaska':
        # AK omits nst/gap/horizontal error for many origins. Preserve the
        # missing fields; do not assert that these pass the spatial screen.
        keep=common & f.rms.notna() & (f.rms<=1) & f.depth_error_km.notna() & (f.depth_error_km<=5)
        policy='manual reviewed earthquake; M>=3; RMS<=1s; depth error<=5km. nst/gap/horizontal error may be missing; no high horizontal-precision claim.'
    elif provider=='SED_Switzerland':
        keep=common & spatial & (f.station_count>=8)
        policy='preferred QuakeML origin manual confirmed/reviewed/final; earthquake only; M>=2; stations>=8; gap<=180; RMS<=1s; horizontal error<=2km; depth error<=5km.'
    else:
        keep=common & spatial
        policy=f'manual reviewed earthquake; M>={mc}; stations>=8; gap<=180; RMS<=1s; horizontal error<=2km; depth error<=5km.'
    return keep.fillna(False),spatial,policy


def curate(provider,frame,paths):
    f=frame.copy()
    for c in ['latitude','longitude','depth_km','magnitude','station_count','azimuthal_gap','rms','horizontal_error_km','depth_error_km']:
        f[c]=pd.to_numeric(f[c],errors='coerce')
    valid=(f.time.notna()&f.event_id.notna()&np.isfinite(f[['latitude','longitude','depth_km','magnitude']]).all(axis=1)
           &f.latitude.between(-90,90)&f.longitude.between(-180,180))
    la,lb,lo,lp=SOURCES[provider]['box']
    wrapped_lon=((f.longitude-lo)%360)+lo
    in_box=f.latitude.between(la,lb)&wrapped_lon.between(lo,lp)
    report=dict(provider=provider,input_rows=len(f),invalid_required_rows=int((~valid).sum()),
                outside_preferred_origin_box=int((valid&~in_box).sum()),
                input_event_types=f.event_type.fillna('missing').value_counts().to_dict(),
                raw_manifest=[m for p,m in paths],source=SOURCES[provider])
    f=f[valid&in_box].copy()
    duplicate=f.event_id.duplicated(keep=False)
    if duplicate.any():
        cols=['event_id','time','magnitude','latitude','longitude','depth_km']
        if f.loc[duplicate,cols].drop_duplicates().event_id.duplicated().any():
            raise ValueError('Conflicting repeated event IDs')
    before=len(f);f=f.drop_duplicates('event_id').sort_values(['time','event_id'],kind='stable').reset_index(drop=True)
    report['identical_repeated_ids_removed']=before-len(f)
    f['quality_screen_pass'],f['spatial_quality_verified'],policy=quality_flags(f,provider)
    f['provider']=provider;f['absolute_time_ready']=True;f['time_basis']='UTC'
    f['same_time_as_previous']=f.time.duplicated()
    out=STORE/provider/'processed';out.mkdir(parents=True,exist_ok=True)
    artifacts={}
    for label,part in [('normalised',f),('quality_screened',f[f.quality_screen_pass])]:
        path=out/f'{label}_20261105.parquet'
        tmp=path.with_suffix('.parquet.part');part.to_parquet(tmp,index=False,compression='zstd');tmp.replace(path)
        artifacts[label]=dict(path=str(path),sha256=sha(path),records=len(part),
            year_counts=part.time.dt.year.value_counts().sort_index().to_dict(),
            magnitude_types=part.mag_type.fillna('missing').value_counts().to_dict())
    report.update(quality_policy=policy,artifacts=artifacts,
        missing_quality_fields={c:int(f[c].isna().sum()) for c in ['station_count','azimuthal_gap','horizontal_error_km','depth_error_km','rms']},
        kept_missing_quality_fields={c:int(f.loc[f.quality_screen_pass,c].isna().sum()) for c in ['station_count','azimuthal_gap','horizontal_error_km']},
        retained_spatial_quality_verified=int(f.loc[f.quality_screen_pass,'spatial_quality_verified'].sum()))
    pd.DataFrame([dict(year=y,records=len(g),reviewed_earthquakes=int((g.event_type.eq('earthquake')&g.evaluation_mode.eq('manual')).sum()),
        quality_screened=int(g.quality_screen_pass.sum())) for y,g in f.groupby(f.time.dt.year)]).to_csv(OUT/'shared/reports'/f'{provider}_annual.csv',index=False)
    dump(OUT/'shared/reports'/f'{provider}_audit.json',report)
    log('CURATED',provider,len(f),int(f.quality_screen_pass.sum()))
    return report


def main():
    (OUT/'shared/reports').mkdir(parents=True,exist_ok=True)
    results,errors={},{}
    for provider,cfg in SOURCES.items():
        try:
            log('START',provider)
            paths=[]
            with cf.ThreadPoolExecutor(max_workers=3) as pool:
                jobs=[pool.submit(fetch,provider,datetime(y,1,1),datetime(y+1,1,1)) for y in range(cfg['start'],2026)]
                for j in cf.as_completed(jobs):paths.extend(j.result())
            f=load(provider,sorted(paths,key=lambda v:str(v[0])))
            results[provider]=curate(provider,f,paths)
        except Exception:
            errors[provider]=traceback.format_exc();log('FAILED',provider,errors[provider])
        dump(OUT/'reports/download_progress.json',dict(completed=list(results),errors=errors))
    dump(OUT/'reports/download_summary.json',dict(sources=results,errors=errors))
    if errors:raise SystemExit(1)
    log('ALL_DOWNLOADED')


if __name__=='__main__':main()

