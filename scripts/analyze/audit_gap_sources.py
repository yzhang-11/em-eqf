"""Trace selected coverage anomalies back to the frozen and official sources."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def ncedc(root, out):
    manifest = json.loads((root/"experiments/revision/catalog_expansion_20261105/reports/NCEDC_audit.json").read_text())
    # Find the 2024 raw artifact by its preserved filename and verify its provenance.
    def files(item):
        if isinstance(item, dict):
            if str(item.get("url","")).endswith("/2024.ehpcsv") and "path" in item:
                yield item
            for value in item.values():
                yield from files(value)
        if isinstance(item,list):
            for value in item:
                yield from files(value)
    item, = files(manifest)
    cloud = pd.read_csv(item["path"],encoding="latin1",dtype={"id":"string"})
    cloud["month"] = cloud.time.str[:7]
    cloud["type_display"] = cloud.type.fillna("<missing>").map(lambda x:x.encode("unicode_escape").decode("ascii"))
    pd.crosstab(cloud.month,cloud.type_display).to_csv(out/"ncedc_2024_raw_types.csv")
    url = "https://service.ncedc.org/fdsnws/event/1/query"
    params = dict(start="2024-05-01T00:00:00",end="2024-12-31T23:59:59.999",minmag=2.5,format="text",eventtype="eq")
    response = requests.get(url,params=params,timeout=45)
    response.raise_for_status()
    assert response.text.lstrip().startswith("#EventID")
    (out/"ncedc_2024_fdsn.txt").write_bytes(response.content)
    fdsn = pd.read_csv(io.StringIO(response.text),sep="|",dtype=str)
    fdsn.columns = fdsn.columns.str.strip()
    fdsn = fdsn.rename(columns={"#EventID":"id","Time":"fdsn_time","Latitude":"fdsn_lat",
        "Longitude":"fdsn_lon","Magnitude":"fdsn_mag","EventType":"fdsn_type"})
    assert fdsn.fdsn_type.str.strip().eq("earthquake").all()
    aligned = cloud.merge(fdsn[["id","fdsn_time","fdsn_lat","fdsn_lon","fdsn_mag","fdsn_type"]],on="id",validate="one_to_one")
    dt = (pd.to_datetime(aligned.time,utc=True,format="mixed")-
          pd.to_datetime(aligned.fdsn_time,utc=True,format="mixed")).abs().dt.total_seconds()
    lat1, lon1 = np.radians(aligned.latitude), np.radians(aligned.longitude)
    lat2, lon2 = np.radians(aligned.fdsn_lat.astype(float)), np.radians(aligned.fdsn_lon.astype(float))
    h = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    km = 12742*np.arcsin(np.sqrt(np.clip(h,0,1)))
    bad = ~aligned.type.eq("eq") & ~aligned.type.eq("earthquake")
    corroborated = bad & dt.le(2) & km.le(15)
    quality = (aligned.mag.ge(2.5) & aligned.nst.ge(8) & aligned.gap.between(0,180)
               & aligned.rms.between(0,.5) & aligned.horizontalError.between(0,2)
               & aligned.depthError.between(0,5))
    aligned["time_difference_s"] = dt
    aligned["distance_km"] = km
    aligned["passes_existing_numeric_quality"] = quality
    aligned.loc[corroborated,["id","time","latitude","longitude","depth","mag","type_display","fdsn_type",
        "fdsn_time","time_difference_s","distance_km","passes_existing_numeric_quality"]].to_csv(
            out/"ncedc_2024_type_recovery_candidates.csv",index=False)
    return dict(query_url=response.url,cloud_path=item["path"],cloud_url=item["url"],
        fdsn_events=len(fdsn),cloud_id_matches=len(aligned),
        matched_original_types=aligned.type_display.value_counts().to_dict(),
        bad_type_and_time_space_correspondence=int(corroborated.sum()),
        repair_candidates_pass_numeric_quality=int((corroborated & quality).sum()),
        fdsn_events_by_month=fdsn.fdsn_time.str[:7].value_counts().sort_index().to_dict())


def montana():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/count"
    common = dict(minlatitude=43,maxlatitude=50,minlongitude=-117,maxlongitude=-104,
                  minmagnitude=1.999,eventtype="earthquake",reviewstatus="reviewed")
    result = []
    for lo,hi in [("2022-08-01","2023-03-31T23:59:59.999"),("2025-11-01","2025-12-31T23:59:59.999")]:
        for contributor in ("mb",None):
            query = common | dict(starttime=lo,endtime=hi)
            if contributor:
                query["contributor"] = contributor
            response=requests.get(url,params=query,timeout=25)
            response.raise_for_status()
            result.append(dict(contributor=contributor,start=lo,end=hi,count=int(response.text),url=response.url))
    return result


def local_sources(store, out):
    gaps=pd.read_csv(out/"coverage_gaps.csv")
    row=gaps[gaps.dataset.eq("Taiwan_CWA")].iloc[0]
    cwa=pd.read_parquet(store/"source_expansion_20261106/Taiwan_CWA/processed/normalized_all.parquet")
    interval=cwa.loc[cwa.time.gt(row.start)&cwa.time.lt(row.end)]
    santorini=pd.read_csv(store/"doi_expansion_20261106/raw/santorini.csv",skipinitialspace=True)
    day=santorini["date-time"].str[:10]
    quality=(santorini.Nphs.ge(12)&santorini.Gap.between(0,180)&santorini.RMS.between(0,.35)
             &santorini.errH.between(0,2)&santorini.errZ.between(0,5)&santorini.Mamp.ge(1.5))
    daily=[dict(day=d,source=int(day.eq(d).sum()),quality_at_cutoff=int((day.eq(d)&quality).sum()))
           for d in ("2025-02-02","2025-02-03","2025-02-04","2025-02-05")]
    return dict(CWA_longest_standard_gap=dict(start=row.start,end=row.end,source=len(interval),
        review_codes=interval.source_review_status.value_counts().to_dict()),Santorini_daily=daily)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("/root/autodl-tmp/em_eqf"))
    parser.add_argument("--store",type=Path,default=Path("/root/autodl-fs/em_eqf_catalogs"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=local_sources(args.store,args.output)
    result["NCEDC"]=ncedc(args.root,args.output)
    result["Montana"]=montana()
    (args.output/"source_gap_checks.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
