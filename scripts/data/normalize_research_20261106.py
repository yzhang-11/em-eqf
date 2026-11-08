"""Normalize DOI-versioned research catalogs with source-specific quality rules.

Research catalogs retain unreported review/station fields. Published preferred
solutions are never mixed with the quality fields of an alternative origin.
"""
import concurrent.futures as cf
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import warnings
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from obspy.io.nordic.core import read_nordic

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
STORE=Path('/root/autodl-fs/em_eqf_catalogs/research_20261106')
OUT=ROOT/'experiments/revision/catalog_research_20261106'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.part');temp.write_text(json.dumps(value,indent=2,default=str,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def extract(name,archive,filename):
    path=STORE/name/'raw'/archive
    meta=json.loads(path.with_name(path.name+'.json').read_text());assert sha(path)==meta['sha256']
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if Path(n).name==filename and not n.startswith('__MACOSX/')]
        assert len(names)==1
        data=z.read(names[0])
    out=STORE/name/'raw/unpacked'/filename;out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():assert out.read_bytes()==data
    else:out.write_bytes(data)
    return out,dict(path=str(path),sha256=meta['sha256'],member=names[0],member_path=str(out),member_sha256=sha(out),**{k:meta[k] for k in ['doi','dsid','license','url']})


def num(value):return float(value) if value is not None else np.nan
def field(node,path):return node.findtext(path,default='') if node is not None else ''
def number(node,path,scale=1):
    value=field(node,path);return float(value)/scale if value.strip() else np.nan


def screen_sw(f):
    return (f.event_type.eq('earthquake')&f.mag_type.eq('ML')&(f.magnitude>=0)&
        (f.station_count>=8)&(f.azimuthal_gap<=180)&(f.rms<=1)&
        (f.horizontal_error_km>0)&(f.horizontal_error_km<=2)&
        (f.depth_error_km>0)&(f.depth_error_km<=5)&
        f.depth_indicator.eq('')&f.location_indicator.eq('')&~f.time_fixed).fillna(False)


def screen_bit(f):
    return (f.event_type.eq('earthquake')&f.mag_type.eq('ML')&(f.magnitude>=0)&
        (f.horizontal_error_km>0)&(f.horizontal_error_km<=2)&
        (f.depth_error_km>0)&(f.depth_error_km<=5)&
        f.preferred_origin_found&f.preferred_magnitude_found).fillna(False)


def sw_frame():
    path,provenance=extract('NORWAY_SW','marck_1.zip','earthquake_catalogue_SW_Norway_hypocenter_location.txt')
    blocks=[];block=[]
    for line in path.read_text(encoding='latin-1').splitlines():
        if not line.strip():
            if block:blocks.append(block);block=[]
        else:block.append(line)
    if block:blocks.append(block)
    with warnings.catch_warnings(record=True) as notes:
        catalog=read_nordic(str(path),nordic_format='NEW')
    assert len(catalog)==len(blocks)==1026
    rows=[]
    for index,(event,block) in enumerate(zip(catalog,blocks),1):
        header=block[0].ljust(80);o=event.origins[0]
        # Nordic main header is the primary solution, not an arbitrary fallback.
        mags=[m for m in event.magnitudes if m.magnitude_type=='ML' and str(m.origin_id)==str(o.resource_id)]
        mags.sort(key=lambda m:0 if m.creation_info and m.creation_info.agency_id=='BER' else 1)
        mag=mags[0] if mags else None
        q=o.quality;u=o.origin_uncertainty
        rows.append(dict(event_id=f'NORWAY_SW:nordic:{index:06d}',raw_record_index=index,
            time=str(o.time),latitude=o.latitude,longitude=o.longitude,depth_km=num(o.depth)/1000,
            magnitude=num(mag.mag) if mag else np.nan,mag_type='ML' if mag else '',
            magnitude_agency=mag.creation_info.agency_id if mag and mag.creation_info else '',
            event_type=event.event_type or '',evaluation_mode=o.evaluation_mode or '',evaluation_status=o.evaluation_status or '',
            station_count=num(getattr(q,'used_station_count',None)),azimuthal_gap=num(getattr(q,'azimuthal_gap',None)),
            rms=num(getattr(q,'standard_error',None)),horizontal_error_km=num(getattr(u,'max_horizontal_uncertainty',None))/1000,
            depth_error_km=num(o.depth_errors.uncertainty)/1000,depth_indicator=header[43].strip(),
            location_indicator=header[44].strip(),time_fixed=bool(o.time_fixed),
            source_agency=o.creation_info.agency_id if o.creation_info else '',
            origin_selection='Nordic primary header; ML linked to primary origin',
            horizontal_error_basis='ObsPy ellipse reconstructed from Nordic E-line covariance, maximum semi-axis',
            manual_review_verified=False,absolute_time_ready=True,time_basis='Nordic/SEISAN UTC origin-time convention',raw_file=str(path)))
    f=pd.DataFrame(rows)
    f['quality_screen_pass']=screen_sw(f)
    policy='Research primary Nordic solution: explicit earthquake; primary-origin ML>=0; stations>=8; gap<=180; RMS<=1s; formal maximum horizontal ellipse axis in (0,2]km and depth error in (0,5]km; no fixed-depth/location/time flags. Review status unreported, not fabricated as manual. Formal ellipse derived by ObsPy from Nordic covariance; confidence level not assumed equivalent to USGS.'
    return f,provenance,policy,dict(parser_warnings=sorted(set(str(w.message) for w in notes)),review_status_not_reported=True)


def bit_event(e,path):
    oid=field(e,'{*}preferredOriginID');mid=field(e,'{*}preferredMagnitudeID')
    origin=next((v for v in e.findall('{*}origin') if v.get('publicID')==oid),None)
    mag=next((v for v in e.findall('{*}magnitude') if v.get('publicID')==mid),None)
    return dict(event_id=e.get('publicID'),time=field(origin,'{*}time/{*}value'),
        latitude=number(origin,'{*}latitude/{*}value'),longitude=number(origin,'{*}longitude/{*}value'),
        depth_km=number(origin,'{*}depth/{*}value',1000),magnitude=number(mag,'{*}mag/{*}value'),
        mag_type=field(mag,'{*}type'),event_type=field(e,'{*}type'),
        preferred_origin_found=origin is not None,preferred_magnitude_found=mag is not None,
        evaluation_mode=field(origin,'{*}evaluationMode'),evaluation_status=field(origin,'{*}evaluationStatus'),
        station_count=number(origin,'{*}quality/{*}usedStationCount'),azimuthal_gap=number(origin,'{*}quality/{*}azimuthalGap'),
        rms=number(origin,'{*}quality/{*}standardError'),
        horizontal_error_km=number(origin,'{*}originUncertainty/{*}horizontalUncertainty',1000),
        depth_error_km=number(origin,'{*}depth/{*}uncertainty',1000),
        source_agency=field(origin,'{*}creationInfo/{*}agencyID'),solution_author=field(origin,'{*}creationInfo/{*}author'),
        origin_selection='Source preferredOriginID only; no alternative-origin quality substitution',
        horizontal_error_basis='Reported preferred-origin horizontalUncertainty, meters to kilometers',
        manual_review_verified=False,absolute_time_ready=True,time_basis='Source QuakeML UTC Z timestamps',raw_file=str(path))


def bit_frame():
    path,provenance=extract('NORWAY_Bitdalsvatn','halpaap_1.zip','bitdalsvatnet_catalog.xml')
    events=ET.parse(path).findall('.//{*}event')
    f=pd.DataFrame([bit_event(e,path) for e in events]);assert len(f)==1593
    f['quality_screen_pass']=screen_bit(f)
    policy='Research preferred high-accuracy origin as documented by author README (GrowClust when available): explicit earthquake; preferred ML>=0; reported preferred-origin horizontal uncertainty in (0,2]km and depth uncertainty in (0,5]km. Station/gap/RMS and review states may be absent and remain unreported; not equivalent to the operational reviewed/station/gap/RMS screen. No quality fields borrowed from alternative Bayesloc/HYPOCENTER origins.'
    return f,provenance,policy,dict(author_readme_preferred_high_accuracy=True,
        published_method='Template matching with high-resolution relative magnitudes and relocations; DOI 10.1785/0220240054')


def sw_relocated_frame():
    absolute,absolute_source,_,_=sw_frame()
    nordic_path=Path(absolute.raw_file.iloc[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');catalog=read_nordic(str(nordic_path),nordic_format='NEW')
    path,relocation_source=extract('NORWAY_SW','marck_1.zip','hypoDD_relocation.txt')
    names='ID LAT LON DEPTH X Y Z EX EY EZ YR MO DY HR MI SC MAG NCCP NCCS NCTP NCTS RCC RCT CID'.split()
    data=pd.read_csv(path,sep=r'\s+',skiprows=1,names=names)
    assert len(data)==684 and not data.ID.duplicated().any()
    rows=[]
    for v in data.itertuples():
        assert 1<=v.ID<=len(catalog)
        event=catalog[int(v.ID)-1];o=event.origins[0]
        time=pd.Timestamp(year=int(v.YR),month=int(v.MO),day=int(v.DY),hour=int(v.HR),minute=int(v.MI),tz='UTC')+pd.to_timedelta(v.SC,unit='s')
        offset=abs((time-pd.Timestamp(str(o.time))).total_seconds())
        assert offset<=5,'HypoDD ID/ordinal link is inconsistent with source Nordic origin time'
        types={mag.magnitude_type for mag in event.magnitudes if str(mag.origin_id)==str(o.resource_id)
            and mag.mag is not None and abs(mag.mag-v.MAG)<=1e-6}
        rows.append(dict(event_id=f'NORWAY_SW:hypoDD:{int(v.ID):06d}',raw_record_index=int(v.ID),
            time=time,latitude=v.LAT,longitude=v.LON,depth_km=v.DEPTH,magnitude=v.MAG,
            mag_type='ML' if types=={'ML'} else '|'.join(sorted(types)),
            event_type=event.event_type or '',evaluation_mode='',evaluation_status='',
            station_count=np.nan,azimuthal_gap=np.nan,rms=np.nan,horizontal_error_km=np.nan,depth_error_km=np.nan,
            catalog_differential_count=int(v.NCTP+v.NCTS),crosscorrelation_differential_count=int(v.NCCP+v.NCCS),
            catalog_differential_rms=v.RCT if v.RCT>=0 else np.nan,
            crosscorrelation_differential_rms=v.RCC if v.RCC>=0 else np.nan,
            reported_hypodd_EX_m=v.EX,reported_hypodd_EY_m=v.EY,reported_hypodd_EZ_m=v.EZ,
            uses_hypodd_formal_errors=False,absolute_time_ready=True,manual_review_verified=False,
            time_basis='HypoDD origin matched to source Nordic UTC record; verified ID/ordinal time offset <=5s',
            identity_link_basis='Inferred integer ID to Nordic record ordinal, validated for all 684 events; no author crosswalk supplied',
            original_nordic_time=str(o.time),relocation_time_offset_seconds=offset,
            source_agency='Author research catalog',raw_file=str(path),absolute_reference_file=str(nordic_path)))
    f=pd.DataFrame(rows)
    ct=(f.catalog_differential_count>=8)&f.catalog_differential_rms.between(0,.5)
    cc=(f.crosscorrelation_differential_count>=8)&f.crosscorrelation_differential_rms.between(0,.5)
    f['quality_screen_pass']=f.event_type.eq('earthquake')&f.mag_type.eq('ML')&(f.magnitude>=0)&(ct|cc)
    policy='Research relative relocation: author HypoDD solution; explicit source earthquake; ML type linked unambiguously to primary Nordic magnitude with exact numeric match; ML>=0; >=8 catalog or cross-correlated differential-time constraints with corresponding differential RMS<=0.5s. These counts are constraints, not stations, and differential RMS is not an absolute-origin RMS. EX/EY/EZ are preserved but not screened as validated uncertainty. No absolute <=2km accuracy or manual-review equivalence is claimed.'
    auxiliary={}
    for label,part in [('absolute_nordic_normalised',absolute),('absolute_nordic_precision_subset',absolute[absolute.quality_screen_pass])]:
        target=STORE/'NORWAY_SW/processed'/f'{label}_20261106.parquet';target.parent.mkdir(parents=True,exist_ok=True)
        part.to_parquet(target,index=False,compression='zstd');auxiliary[label]=dict(path=str(target),sha256=sha(target),records=len(part))
    return f,relocation_source,policy,dict(absolute_source=absolute_source,auxiliary_artifacts=auxiliary,
        absolute_and_relative_are_same_events=True,maximum_link_time_offset_seconds=float(f.relocation_time_offset_seconds.max()),
        identity_link_inferred_and_validated=True)


def publish(name,f,provenance,policy,extra):
    f['time']=pd.to_datetime(f.time,format='mixed',utc=True,errors='coerce').dt.as_unit('ns')
    valid=f.time.notna()&f.event_id.notna()&np.isfinite(f[['latitude','longitude','depth_km','magnitude']]).all(axis=1)
    valid &= f.latitude.between(-90,90)&f.longitude.between(-180,180)
    if f.event_id.duplicated().any():raise ValueError('Conflicting/duplicate research IDs require review')
    f['provider']=name;f['quality_protocol']='research-source-specific-20261106-v1'
    f['quality_policy']=policy;f['regional_strict_equivalent']=False
    f['quality_screen_pass'] &= valid
    f=f.sort_values(['time','event_id'],kind='stable').reset_index(drop=True)
    out=STORE/name/'processed';out.mkdir(parents=True,exist_ok=True)
    artifacts={}
    for label,part in [('normalised',f),('quality_screened',f[f.quality_screen_pass])]:
        path=out/f'{label}_20261106.parquet';part.to_parquet(path,index=False,compression='zstd')
        artifacts[label]=dict(path=str(path),sha256=sha(path),records=len(part),
            year_counts=part.time.dt.year.value_counts().sort_index().to_dict())
    result=dict(provider=name,source=dict(reference=f'https://www.isc.ac.uk/dataset_repository/view_submission.php?dsid={provenance["dsid"]}',
        doi=provenance['doi'],license=provenance['license']),raw_manifest=[provenance],quality_policy=policy,
        quality_protocol='research-source-specific-20261106-v1',regional_strict_equivalent=False,
        input_rows=len(f),invalid_required_rows=int((~valid).sum()),artifacts=artifacts,
        kept_missing_fields={c:int(f.loc[f.quality_screen_pass,c].isna().sum()) for c in ['station_count','azimuthal_gap','rms']},**extra)
    dump(OUT/'reports'/f'{name}_audit.json',result)
    print('NORMALISED',name,len(f),'SCREENED',int(f.quality_screen_pass.sum()),artifacts['quality_screened']['year_counts'],flush=True)
    return result


def stage_auxiliary():
    path,provenance=extract('NORWAY_Sorfjorden','marck_2.zip','hypoDD_sorfjorden_catalogue_simplified.csv')
    f=pd.read_csv(path).rename(columns={'Date':'time','Lon':'longitude','Lat':'latitude','Depth':'depth_km','Mag':'magnitude'})
    f['event_id']=[f'NORWAY_Sorfjorden:row:{i+1:06d}' for i in range(len(f))]
    f['absolute_time_ready']=False;f['quality_screen_pass']=False
    f['time_basis']='Source UTC offset not explicitly documented in CSV/readme'
    f['uncertainty_note']='Author README says HypoDD EX/EY/EZ are not meaningful; not used for filtering'
    p=STORE/'NORWAY_Sorfjorden/processed/normalised_staged_20261106.parquet';p.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(p,index=False)
    sorf=dict(dataset='NORWAY_Sorfjorden',records=len(f),path=str(p),sha256=sha(p),raw_manifest=[provenance],
        reason='Absolute UTC basis not independently confirmed; reported HypoDD EX/EY/EZ explicitly unsuitable as quality estimates',ready=False)
    path,provenance=extract('CHILE_Maule','lieser_1.zip','Catalogue_MAULE_092010-122010_Lieser.dat')
    f=pd.read_csv(path,sep=r'\s+');f=f.rename(columns={'LAT':'latitude','LONG':'longitude','DEPTH':'depth_km'})
    f['magnitude']=np.nan;f['absolute_time_ready']=False;f['quality_screen_pass']=False
    f['location_quality_A']=f.QUALITY.eq('A');f['location_quality_ABC']=f.QUALITY.isin(['A','B','C'])
    p=STORE/'CHILE_Maule/processed/location_only_20261106.parquet';p.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(p,index=False)
    maule=dict(dataset='CHILE_Maule',records=len(f),quality_A=int(f.location_quality_A.sum()),
        quality_ABC=int(f.location_quality_ABC.sum()),path=str(p),sha256=sha(p),raw_manifest=[provenance],
        reason='Source has no magnitude column; retained for location-only studies, not current time-magnitude model',ready=False)
    result=[sorf,maule];dump(OUT/'reports/staged_sources.json',result);return result


def montana():
    name='USGS_Montana';audit=OUT/'reports'/f'{name}_audit.json'
    if audit.exists():
        result=json.loads(audit.read_text())
        for item in result['artifacts'].values():assert sha(item['path'])==item['sha256']
        return result
    spec=importlib.util.spec_from_file_location('_research_regional_shared',ROOT/'scripts/data/download_more_catalogs.py')
    dl=importlib.util.module_from_spec(spec);spec.loader.exec_module(dl)
    cfg=dict(kind='usgs',url='https://earthquake.usgs.gov/fdsnws/event/1/query',contributor='mb',
        box=(43.,50.,-117.,-104.),mc=2.,fetch_minmagnitude=1.999,start='2014-01-01',end='2026-01-01',
        reference='https://earthquake.usgs.gov/data/comcat/catalog/mb/')
    dl.SOURCES[name]=cfg;dl.STORE=STORE;dl.OUT=OUT
    dl.shared.SOURCES=dl.SOURCES;dl.shared.STORE=STORE;dl.shared.OUT=OUT
    paths=[]
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        for part in pool.map(lambda p:dl.fetch(name,*p),dl.periods(cfg)):paths.extend(part)
    paths=sorted(paths,key=lambda p:str(p[0]));frame=dl.shared.load(name,paths)
    result=dl.shared.curate(name,frame,paths)
    for label,item in result['artifacts'].items():
        target=Path(item['path']).with_name(f'{label}_20261106.parquet')
        if target.exists():assert sha(target)==item['sha256']
        else:target.write_bytes(Path(item['path']).read_bytes())
        item['path']=str(target)
    result['quality_protocol']='regional-reviewed-strict-20261106-v1';result['regional_strict_equivalent']=True
    dump(audit,result);return result


def main():
    (OUT/'reports').mkdir(parents=True,exist_ok=True)
    results={}
    for name,fn in [('NORWAY_SW',sw_relocated_frame),('NORWAY_Bitdalsvatn',bit_frame)]:
        results[name]=publish(name,*fn())
    stage_auxiliary()
    results['USGS_Montana']=montana()
    dump(OUT/'reports/download_summary.json',dict(sources=results,errors={}))


if __name__=='__main__':main()
