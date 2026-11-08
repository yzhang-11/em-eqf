"""Versioned earthquake-catalog expansion; never changes existing experiments.

Run with the project's existing pandas/pyarrow/requests environment. Raw HTTP
responses, request parameters, checksums and filtering losses are retained.
Quality cuts are screening rules, not estimates of magnitude completeness.
"""
import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import re
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

STAMP = '20261105'
SOURCES = {
    'JMA': 'https://www.data.jma.go.jp/eqev/data/bulletin/hypo_e.html',
    'GeoNet': 'https://www.geonet.org.nz/data/types/eq_catalogue',
    'INGV': 'https://data.ingv.it/en/dataset/09',
    'NCEDC': 'https://ncedc.org/db/cloud.html',
    'IPOCDeep': 'https://doi.org/10.5880/fidgeo.2026.050',
}
POLICIES = {
    'JMA': 'Agency J; natural event 1; determination K or k; location precision 1/2/3/4/5; M>=2.0.',
    'GeoNet': 'Explicit earthquake; manual; status confirmed/reviewed/final; >=5 stations; gap<=180 deg; RMS<=1 s; M>=3.0.',
    'INGV': 'Explicit earthquake; preferred QuakeML origin manual and reviewed/final; M>=2.5. Quality metadata joined by event ID.',
    'NCEDC': 'Explicit eq/earthquake; >=8 stations; gap<=180 deg; RMS<=0.5 s; horizontal error<=2 km; depth error<=5 km; M>=2.0. Numerical location screen, not proof of manual review.',
    'IPOCDeep': 'Source quality A/B and M>=2.3. Staging only: source timestamps have no explicit UTC offset.',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(path)


def log(*args):
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'), *args, flush=True)


def get(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=(15, 90))
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            return r
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def http_file(storage, provider, label, url, params=None, kind='csv'):
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:16]
    path = storage / provider / 'raw' / f'{label}_{key}.{kind}'
    meta = path.with_name(path.name + '.json')
    if path.exists() and meta.exists():
        m = json.loads(meta.read_text())
        if sha(path) != m['sha256']:
            raise ValueError(f'Cached response checksum changed: {path}')
        return path, m
    r = get(url, params)
    if r.status_code == 413:
        return None, {'split_required': True, 'url': r.url}
    if r.status_code != 204:
        r.raise_for_status()
    content = r.content if r.status_code != 204 else b''
    if content and content.lstrip().startswith((b'<html', b'<!DOCTYPE html')):
        raise ValueError(f'HTML response instead of catalog: {r.url}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    tmp.write_bytes(content)
    tmp.replace(path)
    m = dict(url=r.url, request_params=params, retrieved_utc=datetime.now(timezone.utc).isoformat(),
             http_status=r.status_code, bytes=len(content), sha256=sha(path),
             etag=r.headers.get('ETag'), last_modified=r.headers.get('Last-Modified'), path=str(path))
    dump(meta, m)
    return path, m


def period(storage, provider, start, end, xml=False, depth=0):
    label = start.strftime('%Y%m%dT%H%M%S') + '_' + end.strftime('%Y%m%dT%H%M%S')
    if provider == 'GeoNet':
        url = 'https://quakesearch.geonet.org.nz/csv'
        params = dict(bbox='165,-48,180,-33', startdate=start.isoformat(), enddate=end.isoformat())
    else:
        url = 'https://webservices.ingv.it/fdsnws/event/1/query'
        params = dict(starttime=start.isoformat(), endtime=end.isoformat(), minlatitude=35,
                      maxlatitude=48, minlongitude=6, maxlongitude=19,
                      eventtype='earthquake', format='xml' if xml else 'text')
        if xml:
            params['minmagnitude'] = 2.499
    path, m = http_file(storage, provider, label + ('_reviewed' if xml else ''), url, params,
                        kind='xml' if xml else 'csv' if provider == 'GeoNet' else 'txt')
    n = 0
    if path is not None and m['bytes']:
        if xml:
            n = len(ET.parse(path).findall('.//{*}event'))
        else:
            with path.open() as f:
                n = max(0, sum(1 for _ in f) - 1)
        if provider == 'GeoNet' and n >= 19000:
            m['split_required'] = True
        if provider == 'INGV' and n >= 19000:
            m['split_required'] = True
    if m.get('split_required'):
        if depth >= 18 or (end-start).total_seconds() <= 1:
            raise ValueError(f'Cannot safely subdivide saturated query {label}')
        middle = start + (end-start)/2
        middle = middle.replace(microsecond=0)
        return period(storage, provider, start, middle, xml, depth+1) + period(storage, provider, middle, end, xml, depth+1)
    m = {**m, 'raw_rows': n, 'start': start.isoformat(), 'end_exclusive': end.isoformat()}
    log('downloaded', provider, label, 'xml' if xml else 'text', n)
    return [(path, m)]


def download(storage, provider, workers):
    results = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        if provider == 'NCEDC':
            tasks = [pool.submit(http_file, storage, provider, str(y),
                     f'https://ncedc-pds.s3.us-east-2.amazonaws.com/earthquake_catalogs/NCEDC/{y}.ehpcsv')
                     for y in range(1966, 2026)]
        else:
            tasks = [pool.submit(period, storage, provider, datetime(y,1,1), datetime(y+1,1,1))
                     for y in range(1985, 2026)]
        for job in cf.as_completed(tasks):
            v = job.result()
            results.extend(v if isinstance(v, list) else [v])
    return sorted(results, key=lambda x: str(x[0]))


def num(s):
    return pd.to_numeric(s, errors='coerce')


def col(f, key, default=np.nan):
    return f[key] if key in f else pd.Series(default, index=f.index)


def parse_ingv_xml(path):
    rows = []
    if not path.stat().st_size:
        return pd.DataFrame()
    for ev in ET.parse(path).findall('.//{*}event'):
        def text(node, key, default=''):
            return node.findtext(key, default=default) if node is not None else default
        origins, mags = ev.findall('{*}origin'), ev.findall('{*}magnitude')
        oid, mid = text(ev, '{*}preferredOriginID'), text(ev, '{*}preferredMagnitudeID')
        origin = next((o for o in origins if o.attrib['publicID'] == oid), origins[0] if origins else None)
        mag = next((o for o in mags if o.attrib['publicID'] == mid), mags[0] if mags else None)
        eid = re.search(r'eventId=(\d+)', ev.attrib['publicID'], re.I)
        if not eid:
            raise ValueError(f'Unrecognized INGV ID: {ev.attrib["publicID"]}')
        rows.append(dict(event_id=eid.group(1), time=text(origin, '{*}time/{*}value'),
                         event_type=text(ev, '{*}type'), evaluation_mode=text(origin, '{*}evaluationMode'),
                         evaluation_status=text(origin, '{*}evaluationStatus'),
                         magnitude=text(mag, '{*}mag/{*}value'), mag_type=text(mag, '{*}type')))
    return pd.DataFrame(rows)


def load_fdsn(paths, provider):
    frames = []
    for path, m in paths:
        if not m['bytes']:
            continue
        encoding = 'utf-8'
        try:
            f = pd.read_csv(path, sep='|' if provider == 'INGV' else ',', dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            if provider != 'NCEDC':
                raise
            # Some official NCSS snapshots use 0xff 0xff for an unknown type.
            # Latin-1 maps each byte reversibly; these unknown types are then
            # excluded by the explicit earthquake filter, never coerced to eq.
            encoding = 'latin1_byte_preserving'
            f = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='latin1')
            log('byte-preserving NCEDC encoding fallback',path)
        f.columns = f.columns.str.strip()
        for k in f:
            f[k] = f[k].str.strip()
        f = f.rename(columns={
            '#EventID':'event_id','Time':'time','Latitude':'latitude','Longitude':'longitude',
            'Depth/Km':'depth_km','Depth/km':'depth_km','Magnitude':'magnitude','MagType':'mag_type',
            'EventType':'event_type','Author':'author','publicid':'event_id','eventtype':'event_type',
            'origintime':'time','depth':'depth_km','magnitudetype':'mag_type',
            'evaluationmode':'evaluation_mode','evaluationstatus':'evaluation_status',
            'usedstationcount':'station_count','azimuthalgap':'azimuthal_gap','originerror':'rms',
            'id':'event_id','mag':'magnitude','magType':'mag_type','nst':'station_count',
            'gap':'azimuthal_gap','type':'event_type','horizontalError':'horizontal_error_km',
            'depthError':'depth_error_km','status':'source_status'})
        f['raw_file'] = str(path)
        f['raw_text_encoding'] = encoding
        if provider != 'NCEDC':
            t = pd.to_datetime(f.time, format='mixed', utc=True, errors='coerce')
            f = f[(t >= pd.Timestamp(m['start'], tz='UTC')) & (t < pd.Timestamp(m['end_exclusive'], tz='UTC'))]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def jma_mag(raw):
    s = raw.decode('ascii').strip()
    if not s:
        return np.nan
    if s[0] in 'ABC':
        return -(ord(s[0])-ord('A')+1+int(s[1])/10)
    return int(s)/10


def jma_fixed(raw, scale=100):
    # Fortran Fw.d permits trailing spaces for fixed/depth-slice solutions.
    if not raw.strip():
        return np.nan
    if raw[-1:] == b' ':
        return float(raw.strip())
    return int(raw)/scale


def load_jma(root):
    frames, manifests, audit = [], [], Counter()
    for year in range(1998, 2025):
        path = root/'data/JMA/raw'/f'h{year}.zip'
        manifests.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                              url=f'https://www.data.jma.go.jp/eqev/data/bulletin/data/hypo/h{year}.zip', reused_existing=True))
        rows = []
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise ValueError(f'ZIP CRC failure: {path}')
            for member in z.infolist():
                if member.is_dir():
                    continue
                for i, r in enumerate(z.read(member).splitlines()):
                    audit['raw_records'] += 1
                    if len(r) != 96:
                        raise ValueError(f'Unexpected JMA record size {len(r)}')
                    if r[:1] != b'J' or r[60:61] != b'1':
                        audit['non_J_or_non_natural'] += 1
                        continue
                    try:
                        lat = int(r[21:24]) + jma_fixed(r[24:28])/60
                        lon = int(r[32:36]) + jma_fixed(r[36:40])/60
                        sec = jma_fixed(r[13:17])
                        rows.append((f'JMA:{year}:{member.filename}:{i+1}', r[1:13].decode(), sec,
                                     lat, lon, jma_fixed(r[44:49]), jma_mag(r[52:54]),
                                     r[54:55].decode(), r[95:96].decode(), r[59:60].decode(),
                                     int(r[92:95]) if r[92:95].strip() else np.nan,
                                     jma_fixed(r[17:21]), str(path)))
                    except (ValueError, UnicodeError):
                        audit['parse_failure'] += 1
        f = pd.DataFrame(rows, columns=['event_id','minute_jst','seconds','latitude','longitude','depth_km',
                                       'magnitude','mag_type','determination','location_precision','station_count',
                                       'origin_time_error_s','raw_file'])
        f['time'] = (pd.to_datetime(f.pop('minute_jst'), format='%Y%m%d%H%M', errors='coerce', utc=True)
                     + pd.to_timedelta(f.pop('seconds'),unit='s') - pd.Timedelta(hours=9))
        f['event_type'] = 'earthquake'
        f['evaluation_mode'] = np.where(f.determination.isin(['K','k','S','s']), 'manual', 'other')
        frames.append(f)
        log('parsed JMA', year, len(f))
    return pd.concat(frames, ignore_index=True), manifests, dict(audit)


def prepare_frame(frame, provider):
    f = frame.copy()
    f['time'] = pd.to_datetime(f.time, format='mixed', errors='coerce', utc=(provider != 'IPOCDeep'))
    for k in ['latitude','longitude','depth_km','magnitude','station_count','azimuthal_gap','rms',
              'horizontal_error_km','depth_error_km']:
        if k in f:
            f[k] = num(f[k])
    if 'event_type' not in f:
        f['event_type'] = 'earthquake'
    pre = len(f)
    counts = dict(input_rows=pre, event_types=col(f,'event_type','').value_counts(dropna=False).to_dict())
    valid = f.time.notna() & np.isfinite(f.magnitude) & np.isfinite(f.latitude) & np.isfinite(f.longitude)
    valid &= f.latitude.between(-90,90) & f.longitude.between(-180,180)
    counts['invalid_required_fields'] = int((~valid).sum())
    counts['non_earthquake_valid_rows'] = int((valid & ~f.event_type.isin(['eq','earthquake'])).sum())
    f = f[valid & f.event_type.isin(['eq','earthquake'])].copy()
    f = f[(f.time >= '1966-01-01') & (f.time < '2026-01-01')]
    before = len(f)
    f = f.drop_duplicates('event_id')
    counts['duplicate_ids_removed'] = before-len(f)
    # Preserve physically distinct simultaneous events; never jitter their times.
    f = f.sort_values(['time','event_id'],kind='stable').reset_index(drop=True)
    f['provider'] = provider
    f['time_basis'] = 'source_unspecified_offset' if provider == 'IPOCDeep' else 'UTC'
    f['same_time_as_previous'] = f.time.duplicated()
    if provider == 'JMA':
        q = f.determination.isin(['K','k']) & f.location_precision.isin(list('12345')) & (f.magnitude >= 2.0)
    elif provider == 'GeoNet':
        q = ((f.evaluation_mode == 'manual') & f.evaluation_status.isin(['confirmed','reviewed','final'])
             & (f.station_count >= 5) & (f.azimuthal_gap <= 180) & (f.rms <= 1) & (f.magnitude >= 3))
    elif provider == 'INGV':
        q = (col(f,'evaluation_mode','') == 'manual') & col(f,'evaluation_status','').isin(['reviewed','final']) & (f.magnitude >= 2.5)
    elif provider == 'NCEDC':
        q = ((f.station_count >= 8) & (f.azimuthal_gap <= 180) & (f.rms <= .5)
             & (f.horizontal_error_km <= 2) & (f.depth_error_km <= 5) & (f.magnitude >= 2))
    else:
        q = f.quality.isin(['A','B']) & (f.magnitude >= 2.3)
    f['quality_screen_pass'] = q.fillna(False)
    f['absolute_time_ready'] = provider != 'IPOCDeep'
    counts['normalised_earthquakes'] = len(f)
    counts['tied_time_rows_retained'] = int(f.same_time_as_previous.sum())
    return f, counts


def describe(f):
    return dict(records=len(f), start=str(f.time.min()), end=str(f.time.max()),
                magnitude_min=float(f.magnitude.min()) if len(f) else None,
                magnitude_max=float(f.magnitude.max()) if len(f) else None,
                counts_ge_magnitude={str(m):int((f.magnitude>=m).sum()) for m in [1,1.5,2,2.3,2.5,3,4]},
                magnitude_types=col(f,'mag_type','unspecified').value_counts(dropna=False).to_dict(),
                evaluation_modes=col(f,'evaluation_mode','unavailable').value_counts(dropna=False).to_dict(),
                source_status=col(f,'source_status','unavailable').value_counts(dropna=False).to_dict(),
                year_counts=f.time.dt.year.value_counts().sort_index().to_dict())


def save_outputs(storage, reports, provider, frame, counters, manifests):
    f, audit = prepare_frame(frame, provider)
    out = storage/provider/'processed'
    out.mkdir(parents=True,exist_ok=True)
    outputs = []
    for name, part in [('earthquakes',f),('quality_screened', f[f.quality_screen_pass])]:
        path = out/f'{name}_{STAMP}.parquet'
        tmp = path.with_suffix('.parquet.part')
        part.to_parquet(tmp,index=False,compression='zstd')
        tmp.replace(path)
        outputs.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, **describe(part)))
    rows=[]
    for (year, mt), group in f.groupby([f.time.dt.year,col(f,'mag_type','unspecified')],dropna=False):
        counts=group.magnitude.round(1).value_counts().sort_index()
        mode=float(counts.idxmax()) if len(counts) else None
        rows.append(dict(year=int(year),mag_type=str(mt),records=len(group),
                         quality_screened=int(group.quality_screen_pass.sum()),
                         maxc_mode=mode, maxc_plus_0p2_screen_only=None if mode is None else round(mode+.2,1)))
    pd.DataFrame(rows).to_csv(reports/f'{provider}_year_magnitude_screen.csv',index=False)
    audit.update(provider=provider, quality_policy=POLICIES[provider], source=SOURCES[provider],
                 source_audit=counters, outputs=outputs,
                 absolute_time_ready=(provider!='IPOCDeep'), raw_manifest=manifests,
                 notes=['Magnitude types and original precision retained; no conversion to Mw.',
                        'MAXC+0.2 is only a descriptive completeness screen, not a validated Mc.',
                        'Quality/location filtering can change detection completeness; re-estimate Mc per region on training history.',
                        'Same-time distinct events retained; point-process loader must handle ties explicitly.',
                        'No cross-catalog deduplication performed; counts must not be summed as independent events.',
                        'No train/validation/test arrays created; existing paper and transfer pilot unchanged.'])
    dump(reports/f'{provider}_audit.json',audit)
    log('FINISHED',provider, 'normalised',len(f),'quality',int(f.quality_screen_pass.sum()))
    return audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/em_eqf'))
    ap.add_argument('--storage',type=Path,default=Path('/root/autodl-fs/em_eqf_catalogs/expansion_20261105'))
    ap.add_argument('--providers',nargs='+',default=['NCEDC','GeoNet','INGV','JMA','IPOCDeep'])
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    reports=args.root/'experiments/revision/catalog_expansion_20261105/reports'
    reports.mkdir(parents=True,exist_ok=True)
    args.storage.mkdir(parents=True,exist_ok=True)
    link=args.root/'data/CatalogExpansion_20261105'
    if not link.exists() and not link.is_symlink():
        link.symlink_to(args.storage,target_is_directory=True)
    protected=[args.root/'data/MultiRegion/catalogs/pilot_2007_2019.pt',
               args.root/'scripts/data/prepare_multiregion_pretraining.py',
               args.root/'scripts/data/inventory_pretraining_catalogs.py']
    baseline={str(p):sha(p) for p in protected}
    dump(reports/'protected_baseline.json',baseline)
    audits,errors={},{}
    for provider in args.providers:
        try:
            log('START',provider)
            counters={}
            if provider=='JMA':
                frame,manifests,counters=load_jma(args.root)
            elif provider=='IPOCDeep':
                path=args.root/'data/IPOCDeep/raw/2026-050_Najafipour-et-al_hypocenters.csv'
                frame=pd.read_csv(path,skiprows=3)
                frame=frame.rename(columns={'depth':'depth_km'})
                frame['event_id']=['IPOCDeep:'+str(i+1) for i in range(len(frame))]
                frame['raw_file']=str(path)
                manifests=[dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,reused_existing=True)]
            else:
                paths=download(args.storage,provider,args.workers)
                manifests=[m for _,m in paths]
                frame=load_fdsn(paths,provider)
                if provider=='INGV':
                    xml=[]
                    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
                        jobs=[pool.submit(period,args.storage,provider,datetime(y,1,1),datetime(y+1,1,1),True) for y in range(1985,2026)]
                        for job in cf.as_completed(jobs):
                            xml.extend(job.result())
                    detailed=pd.concat([parse_ingv_xml(p) for p,m in xml],ignore_index=True)
                    detailed=detailed.drop_duplicates('event_id')
                    # Keep text magnitude authoritative; join the preferred-origin review flags.
                    detail=detailed.loc[detailed.event_type=='earthquake', ['event_id','evaluation_mode','evaluation_status']]
                    frame=frame.merge(detail,on='event_id',how='left',validate='many_to_one')
                    manifests.extend(m for p,m in xml)
                    counters['xml_events']=len(detailed)
                    counters['xml_event_types']=detailed.event_type.value_counts(dropna=False).to_dict()
            audits[provider]=save_outputs(args.storage,reports,provider,frame,counters,manifests)
        except Exception:
            errors[provider]=traceback.format_exc()
            log('ERROR',provider,errors[provider])
        dump(reports/'progress.json',dict(completed=list(audits),errors=errors,storage=str(args.storage)))
    after={str(p):sha(p) for p in protected}
    preserved=baseline==after
    dump(reports/'protected_verification.json',dict(unchanged=preserved,before=baseline,after=after))
    dump(reports/'summary.json',dict(date='2026-11-05',storage=str(args.storage),
         audits={k:{field:v[field] for field in ['normalised_earthquakes','quality_policy','absolute_time_ready','outputs']} for k,v in audits.items()},
         errors=errors,protected_unchanged=preserved,code_sha256=sha(Path(__file__))))
    if errors or not preserved:
        raise SystemExit(1)
    log('ALL_COMPLETE',list(audits))


if __name__=='__main__':
    main()

