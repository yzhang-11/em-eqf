"""Resume NOA with audited native exports, adaptive windows and durable coverage.

Reuses old QuakeML snapshots, including the corrected 2019-12 response. New
SC3ML bytes and ObsPy-derived QuakeML both retain checksums and provenance.
"""
import argparse
import concurrent.futures as cf
import fcntl
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET

import obspy
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('_noa_resume_shared',ROOT/'scripts/data/download_more_catalogs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
CFG=m.SOURCES['NOA_Greece']
STORE=m.STORE/'NOA_Greece'
OUT=ROOT/'experiments/revision/catalog_optimization_20261105'
PLANS=STORE/'coverage_native_20261105'
SCHEMA='noa-native-coverage-v1'
SESSIONS=threading.local()
DOWNLOAD_LOCK=None


def acquire_download_lock():
    STORE.mkdir(parents=True,exist_ok=True)
    handle=(STORE/'download_native.lock').open('a+')
    try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close();raise RuntimeError('Another NOA downloader owns this catalog lock')
    return handle


def is_read_timeout(error):
    if isinstance(error,requests.ReadTimeout):return True
    seen=set();current=error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__=='ReadTimeoutError':return True
        current=getattr(current,'__cause__',None) or getattr(current,'__context__',None)
    return False


def http_get(url,params,timeout=(8,90)):
    """Reuse each worker's TLS connection; retry connection faults, not windows."""
    for attempt in range(2):
        if not hasattr(SESSIONS,'session'):SESSIONS.session=requests.Session()
        try:return SESSIONS.session.get(url,params=params,timeout=timeout)
        except (requests.ConnectionError,requests.Timeout) as error:
            if is_read_timeout(error) or attempt:raise
            SESSIONS.session.close();del SESSIONS.session
            time.sleep(2)


def embedded_error(content):
    match=re.search(rb'(?:^|[\r\n])Error (400|413|429|500|502|503|504):',content)
    return int(match.group(1)) if match else None


def preserve_failure(start,end,content,status,code,url,params):
    digest=hashlib.sha256(content).hexdigest()
    path=STORE/'raw/response_failures'/f'{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_http{status}_service{code}_{digest[:12]}.body'
    atomic_bytes(path,content)
    sidecar=path.with_name(path.name+'.json')
    if not sidecar.exists():
        m.shared.dump(sidecar,dict(path=str(path),sha256=digest,bytes=len(content),http_status=status,
            service_error=code,url=url,params=params,retrieved_utc=datetime.now(timezone.utc).isoformat()))
    return str(path)


def params_for(start,end,fmt='xml'):
    a,b,c,d=CFG['box']
    return dict(starttime=start.isoformat(),endtime=(end-timedelta(microseconds=1)).isoformat(),
        minlatitude=a,maxlatitude=b,minlongitude=c,maxlongitude=d,
        minmagnitude=CFG['fetch_minmagnitude'],format=fmt,orderby='time-asc')


def key_for(start,end):
    return hashlib.sha256(json.dumps(params_for(start,end),sort_keys=True).encode()).hexdigest()[:16]


def atomic_bytes(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content:raise ValueError(f'Immutable snapshot differs: {path}')
        return
    temp=path.with_name(path.name+'.part');temp.write_bytes(content);temp.replace(path)


def verify_parts(parts,start,end):
    cursor=start.isoformat()
    for meta in sorted(parts,key=lambda v:v['start']):
        if meta['start']!=cursor:raise ValueError('Coverage has a gap or overlap')
        stop=datetime.fromisoformat(meta['end_exclusive'])
        if not stop>datetime.fromisoformat(cursor):raise ValueError('Nonpositive coverage interval')
        if meta['params']!=params_for(datetime.fromisoformat(cursor),stop):
            raise ValueError('Coverage request parameters differ')
        p=Path(meta['path'])
        if m.shared.sha(p)!=meta['sha256'] or p.stat().st_size!=meta['bytes']:
            raise ValueError(f'Cached response changed: {p}')
        if not meta['bytes'] and not meta.get('empty_verification'):
            raise ValueError('Unverified empty coverage interval')
        if 'native_source' in meta:
            raw=meta['native_source'];path=Path(raw['path'])
            if m.shared.sha(path)!=raw['sha256'] or path.stat().st_size!=raw['bytes']:
                raise ValueError('Native source changed')
            for key,value in meta['params'].items():
                actual=raw['params'][key]
                if key in ('starttime','endtime'):
                    matches=pd.Timestamp(actual)==pd.Timestamp(value)
                elif key=='format':matches=actual=='sc3ml'
                else:matches=actual==value
                if not matches:raise ValueError(f'Native request differs: {key}')
        cursor=meta['end_exclusive']
    if cursor!=end.isoformat():raise ValueError('Coverage does not reach the interval end')
    return sorted(parts,key=lambda v:v['start'])


def plan_path(start,end):
    return PLANS/f'{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_{key_for(start,end)}.json'


def split_window(start,end,reason):
    if (end-start).total_seconds()<=3600:raise ValueError(f'Cannot further split failing hour: {reason}')
    middle=(start+(end-start)/2).replace(microsecond=0)
    children=[dict(start=start.isoformat(),end=middle.isoformat()),dict(start=middle.isoformat(),end=end.isoformat())]
    m.shared.dump(plan_path(start,end),dict(schema=SCHEMA,children=children,reason=reason))
    result=[]
    for child in children:result.extend(fetch_window(datetime.fromisoformat(child['start']),datetime.fromisoformat(child['end'])))
    return publish_plan(start,end,result)


def publish_plan(start,end,parts):
    parts=verify_parts(parts,start,end)
    m.shared.dump(plan_path(start,end),dict(schema=SCHEMA,start=start.isoformat(),
        end_exclusive=end.isoformat(),parts=parts,verified_utc=datetime.now(timezone.utc).isoformat()))
    return parts


def seed_for(start,end):
    path=OUT/'transport_benchmark.json'
    if not path.exists():return None
    for result in json.loads(path.read_text())['results']:
        if result.get('error') or not result.get('events'):continue
        p=result['params']
        if pd.Timestamp(p['starttime'])==pd.Timestamp(start) and pd.Timestamp(p['endtime'])==pd.Timestamp(end)-pd.Timedelta(microseconds=1):
            raw=Path(result['raw'])
            if m.shared.sha(raw)!=result['sha256']:raise ValueError('Benchmark seed changed')
            return result
    return None


def fetch_window(start,end):
    index=plan_path(start,end)
    if index.exists():
        old=json.loads(index.read_text())
        if old.get('schema')!=SCHEMA:raise ValueError('Unexpected coverage plan schema')
        if 'parts' in old:return verify_parts(old['parts'],start,end)
        result=[]
        for c in old['children']:result.extend(fetch_window(datetime.fromisoformat(c['start']),datetime.fromisoformat(c['end'])))
        return publish_plan(start,end,result)
    # The legacy request key is deliberately unchanged.
    legacy=STORE/'raw'/f'{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_{key_for(start,end)}.xml'
    if legacy.exists() and legacy.with_name(legacy.name+'.json').exists():
        return publish_plan(start,end,[meta for _,meta in m.fetch('NOA_Greece',start,end)])

    wire_params=params_for(start,end,'sc3ml');seed=seed_for(start,end)
    stem=f'{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_{key_for(start,end)}'
    raw=STORE/'raw/native_sc3ml'/f'{stem}.sc3ml.xml'
    derived=raw.with_name(f'{stem}.quakeml.xml')
    wire_meta=raw.with_name(raw.name+'.json')
    began=time.monotonic()
    if raw.exists() and wire_meta.exists():
        saved=json.loads(wire_meta.read_text())
        if m.shared.sha(raw)!=saved['sha256']:raise ValueError('Interrupted native cache changed')
        content=raw.read_bytes();status=saved['status'];url=saved['url'];wire_params=saved['params']
        headers=saved.get('headers',{});download_seconds=saved.get('download_seconds',0)
    elif seed:
        content=Path(seed['raw']).read_bytes();status=200;url=seed['url'];wire_params=seed['params']
        headers={};download_seconds=seed['seconds']
    else:
        m.shared.log('NATIVE_REQUEST',start.isoformat(),end.isoformat())
        try:
            r=http_get(CFG['url'],wire_params)
        except (requests.Timeout,requests.ConnectionError) as e:
            if is_read_timeout(e):return split_window(start,end,f'{type(e).__name__}: {e}')
            raise  # DNS/TLS/connect failures are not evidence of excessive data density.
        if r.status_code in (400,413) and re.search(r'limit|maximum|exceed|too large|too many',r.text,re.I):
            return split_window(start,end,f'HTTP {r.status_code}: {r.text[:160]}')
        if r.status_code in (429,500,502,503,504):
            if (end-start).total_seconds()>86400:return split_window(start,end,f'HTTP {r.status_code}')
            time.sleep(2);r=http_get(CFG['url'],wire_params)
        if r.status_code!=204:r.raise_for_status()
        content=b'' if r.status_code==204 else r.content
        status=r.status_code;url=r.url;headers={k:v for k,v in r.headers.items() if k.lower() in ['content-type','content-encoding','content-length']}
        download_seconds=round(time.monotonic()-began,3)
    code=embedded_error(content)
    if code is not None:
        failure=preserve_failure(start,end,content,status,code,url,wire_params)
        if code in (400,413) and re.search(rb'limit|maximum|exceed|too large|too many',content,re.I):
            return split_window(start,end,f'HTTP {status} contains service error {code}; evidence: {failure}')
        raise ValueError(f'HTTP {status} contains embedded service error {code}; evidence: {failure}')
    atomic_bytes(raw,content)
    native=dict(path=str(raw),sha256=m.shared.sha(raw),bytes=len(content),url=url,
        params=wire_params,status=status,retrieved_utc=datetime.now(timezone.utc).isoformat(),
        headers=headers,download_seconds=download_seconds)
    if not wire_meta.exists():m.shared.dump(wire_meta,native)
    else:native=json.loads(wire_meta.read_text())
    empty=None
    if not content or not ET.fromstring(content).findall('.//{*}event'):
        confirm=http_get(CFG['url'],params_for(start,end,'text'))
        if confirm.status_code!=204:confirm.raise_for_status()
        records=[v for v in confirm.text.splitlines() if v.strip() and not v.lstrip().startswith('#') and not v.lower().startswith('eventid|')]
        if records:raise ValueError('Empty native response contradicts text response')
        empty=dict(format='text',status=confirm.status_code,url=confirm.url,
            sha256=hashlib.sha256(confirm.content).hexdigest(),checked_utc=datetime.now(timezone.utc).isoformat())
        atomic_bytes(derived,b'');n=0;conversion_seconds=0.
    else:
        begin=time.monotonic();cat=obspy.read_events(str(raw),format='SC3ML')
        if len(cat)!=len(ET.fromstring(content).findall('.//{*}event')):
            raise ValueError('Conversion dropped events')
        if not derived.exists():
            temp=derived.with_name(derived.name+'.part');cat.write(str(temp),format='QUAKEML');temp.replace(derived)
        parsed=m.parse_quakeml(derived);n=len(parsed)
        if n!=len(cat):raise ValueError('Derived QuakeML changed the event count')
        conversion_seconds=round(time.monotonic()-begin,3)
    meta=dict(path=str(derived),sha256=m.shared.sha(derived),bytes=derived.stat().st_size,
        start=start.isoformat(),end_exclusive=end.isoformat(),params=params_for(start,end),url=url,
        status=status,rows=n,retrieved_utc=datetime.now(timezone.utc).isoformat(),
        artifact_kind='derived_quakeml',native_source=native,
        conversion=dict(library='obspy',version=obspy.__version__,source_format='SC3ML',target_format='QUAKEML',seconds=conversion_seconds))
    if empty is not None:meta['empty_verification']=empty
    m.shared.dump(derived.with_name(derived.name+'.json'),meta)
    parts=publish_plan(start,end,[meta])
    m.shared.log('NATIVE_READY',start.isoformat(),end.isoformat(),n,'download_s',download_seconds)
    return parts


def progress(completed,errors):
    expected=m.periods(CFG)
    missing=[dict(start=a.isoformat(),end_exclusive=b.isoformat()) for a,b in expected if a.isoformat() not in completed]
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),schema=SCHEMA,
        expected_months=len(expected),completed_months=len(completed),missing_months=len(missing),
        ready=not missing and not errors,completed_windows=sorted(completed),missing_windows=missing,errors=errors)
    m.shared.dump(OUT/'coverage_progress.json',result)
    return result


def main():
    global DOWNLOAD_LOCK
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    DOWNLOAD_LOCK=acquire_download_lock()
    completed={};errors={};begin=time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(fetch_window,a,b):(a,b) for a,b in m.periods(CFG)}
        for job in cf.as_completed(jobs):
            a,b=jobs[job]
            try:
                completed[a.isoformat()]=job.result();errors.pop(a.isoformat(),None)
                m.shared.log('MONTH_READY',a.strftime('%Y-%m'),len(completed),'/144')
            except Exception as e:
                errors[a.isoformat()]=f'{type(e).__name__}: {e}'
                m.shared.log('MONTH_FAILED',a.strftime('%Y-%m'),errors[a.isoformat()])
                if len(errors)>=3:
                    for pending in jobs:pending.cancel()
                    progress(completed,errors)
                    raise
            progress(completed,errors)
    state=progress(completed,errors)
    if not state['ready']:raise RuntimeError('Incomplete observation coverage: no standard source published')
    parts=[(Path(meta['path']),meta) for key in sorted(completed) for meta in completed[key]]
    frame=m.shared.load('NOA_Greece',parts)
    audit=m.shared.curate('NOA_Greece',frame,parts)
    manifest=m.OUT/'reports/download_summary.json';current=json.loads(manifest.read_text())
    current['sources']['NOA_Greece']=audit;current.setdefault('errors',{}).pop('NOA_Greece',None)
    current.setdefault('staged_downloads',{}).pop('NOA_Greece',None)
    m.shared.dump(manifest,current)
    state.update(elapsed_seconds=round(time.monotonic()-begin,3),raw_parts=len(parts),
        normalised_records=audit['artifacts']['normalised']['records'],strict_records=audit['artifacts']['quality_screened']['records'],
        source_audit=str(m.OUT/'reports/NOA_Greece_audit.json'))
    m.shared.dump(OUT/'download_complete.json',state)
    m.shared.log('NOA_COMPLETE',state['normalised_records'],state['strict_records'],'elapsed_s',state['elapsed_seconds'])


if __name__=='__main__':main()
