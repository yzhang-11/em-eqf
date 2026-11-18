"""Prepare geographically disjoint regional catalogs paired to reported water injection.
Run inside the project environment with pandas, numpy, shapely 2 and python-calamine.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import intersects_xy
from shapely.geometry import shape
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.preprocessing import estimate_mc_max_curvature

DATA = ROOT / "data"
RAW = DATA / "US_Disposal_Inputs" / "raw"
BARREL_M3 = 0.158987294928
OK_COUNTIES = ("Alfalfa", "Garfield", "Grant", "Logan", "Major", "Noble", "Pawnee", "Payne", "Woods")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def areas_for(longitude, latitude, boundaries, names):
    names = set(names)
    areas = pd.Series(pd.NA, index=longitude.index, dtype="string")
    for feature in boundaries["features"]:
        name = feature["properties"]["NAME"].removesuffix(" County")
        if name not in names:
            continue
        inside = intersects_xy(shape(feature["geometry"]), longitude, latitude)
        if areas[inside].notna().any():
            raise ValueError("Coordinates lie on a shared county boundary")
        areas.loc[inside] = name
    return areas


def oklahoma():
    bounds = json.loads((RAW / "ok_boundaries.json").read_text())
    events, reports = [], []
    for path in sorted(RAW.glob("OGS_archive*.csv")):
        x = pd.read_csv(path, low_memory=False)
        old = "prefmag" in x
        status = x["status"].fillna("archived, status unspecified") if not old else "archived, status unspecified"
        frame = pd.DataFrame({
            "event_id": "ogs:" + path.stem[-4:] + ":row:" + (x.index + 2).astype(str),
            "source_id": x["id" if old else "event_id"].replace(0, pd.NA),
            "time_iso": pd.to_datetime(x.origintime, format="mixed", utc=True),
            "magnitude": x["prefmag" if old else "magnitude"], "magnitude_type": x["pmag_type"] if old else pd.NA,
            "latitude": x.latitude, "longitude": x.longitude, "depth_km": x["depth" if old else "depth_km"],
            "source_status": status, "horizontal_error_lon_km": x.err_lon, "horizontal_error_lat_km": x.err_lat,
        })
        frame["area"] = areas_for(frame.longitude, frame.latitude, bounds, OK_COUNTIES)
        events.append(frame.loc[frame.area.notna() & frame.source_status.isin(["Reviewed", "archived, status unspecified"])])
    cols = ["API", "Report_Date", "Volume_BPD", "Latitude", "Longitude"]
    for path in sorted(RAW.glob("occ*.xlsx")):
        # Stream wide annual workbooks to keep raw-data preparation within 2 GB.
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook.active.iter_rows(values_only=True)
            header = next(rows)
            indices = [header.index(column) for column in cols]
            x = pd.DataFrame.from_records(
                (tuple(row[index] for index in indices) for row in rows), columns=cols)
        finally:
            workbook.close()
        x["API"] = x.API.astype("string").str.replace(r"\.0$", "", regex=True)
        x = x.rename(columns=dict(API="well_id", Report_Date="source_date", Volume_BPD="volume_bbl",
                                  Latitude="latitude", Longitude="longitude"))
        locations = x[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").drop_duplicates()
        locations["area"] = areas_for(locations.longitude, locations.latitude, bounds, OK_COUNTIES)
        x[["latitude","longitude"]] = x[["latitude","longitude"]].apply(pd.to_numeric, errors="coerce")
        x = x.merge(locations.dropna(), on=["latitude", "longitude"], validate="many_to_one")
        reports.append(x)
    eq, inj = pd.concat(events, ignore_index=True), pd.concat(reports, ignore_index=True)
    if inj.groupby("well_id").area.nunique().gt(1).any():
        raise ValueError("An Oklahoma well was assigned to multiple counties")
    for county in OK_COUNTIES:
        yield "Oklahoma_" + county, eq.loc[eq.area == county].drop(columns="area"), inj.loc[inj.area == county], "D"


def kansas():
    x = pd.read_csv(RAW / "kansas_events.csv")
    bounds = json.loads((RAW / "ks_boundary.json").read_text())
    region = areas_for(x.longitude, x.latitude, bounds, ["Harper", "Sumner"])
    eq = x.loc[region.notna() & x.status.eq("reviewed") & x.type.eq("earthquake")].rename(
        columns={"id": "event_id", "time": "time_iso", "mag": "magnitude", "magType": "magnitude_type",
                 "depth": "depth_km", "status": "source_status"}).copy()
    eq.time_iso = pd.to_datetime(eq.time_iso, format="ISO8601", utc=True)
    # The 2015 archive has no permit classification. Use named disposal wells
    # whose complete API and reported coordinates match a 2016 Arbuckle SWD.
    first = pd.read_csv(RAW / "KS_UIC_archive2015.zip", dtype={"API": str}, low_memory=False)
    second = pd.read_csv(RAW / "KS_UIC_archive2016.zip", dtype={"API": str}, low_memory=False)
    permits = second.loc[second.COUNTY.isin(["Harper", "Sumner"]) & second.STATUS.isin(["SWD", "SWD-P&A"])
                         & second.INJECTION_ZONE.str.strip().str.upper().eq("ARBUCKLE"),
                         ["API", "LATITUDE", "LONGITUDE"]].drop_duplicates()
    early = first.loc[first.LEASE.str.contains(r"SWD|DISPOSAL|S\.W\.D", case=False, na=False)]
    early = early.merge(permits, on="API", suffixes=("", "_permit"))
    matched = ((early.LATITUDE - early.LATITUDE_permit).abs() < .001) & ((early.LONGITUDE - early.LONGITUDE_permit).abs() < .001)
    cohort = set(early.loc[matched & early.GAS_INJECTED.eq(0), "API"])
    reports = []
    for path in sorted(RAW.glob("KS_UIC_archive*.zip")):
        x = pd.read_csv(path, dtype={"API": str}, low_memory=False)
        mask = x.API.isin(cohort) & x.MONTH.between(1, 12) & x.GAS_INJECTED.eq(0)
        if "INJECTION_ZONE" in x:
            mask &= x.INJECTION_ZONE.str.strip().str.upper().eq("ARBUCKLE") & x.STATUS.isin(["SWD", "SWD-P&A"])
        x = x.loc[mask]
        reports.append(pd.DataFrame({"well_id": x.API, "source_date": pd.to_datetime(
            {"year": x.YEAR, "month": x.MONTH, "day": 1}), "volume_bbl": x.FLUID_INJECTED}))
    return "Kansas_Harper_Sumner", eq, pd.concat(reports, ignore_index=True), "M"


def delaware():
    x = pd.read_excel(RAW / "tx_catalog.xls", engine="calamine")
    clock = x[["Year", "Month", "Day", "Hour", "Minute"]].rename(columns=str.lower)
    eq = pd.DataFrame({"event_id": "aziz-zanjani:row:" + (x.index + 2).astype(str),
                       "time_iso": pd.to_datetime(clock, utc=True) + pd.to_timedelta(x.Second, unit="s"),
                       "latitude": x["Latitude (degree)"], "longitude": x["Longitude (degree)"],
                       "depth_km": x["Focal Depth (km)"], "magnitude": x.Magnitude, "magnitude_type": pd.NA,
                       "origin_time_error_s": x["Origin Time Uncertainty (sec)"],
                       "depth_uncertainty_km": x["Depth Uncertainty (+/- in km)"]})
    original = pd.read_excel(RAW / "tx_post-2017-injection.xlsx", engine="calamine", dtype=str)
    apis = original["API-Number"].str.replace("-", "", regex=False).str[2:10]
    loc = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(RAW.glob("loc*.csv"))])
    if not loc.api_no.isin(apis).all() or loc.groupby("uic_number").api_no.nunique().gt(1).any():
        raise ValueError("Texas permit to API mapping is ambiguous")
    x = pd.concat([pd.read_csv(f, dtype={"uic_no": str, "type_uic": str}) for f in sorted(RAW.glob("h10*.csv"))])
    if not x.uic_no.isin(loc.uic_number).all() or x.id.duplicated().any():
        raise ValueError("Unexpected or duplicate H-10 filing")
    # H-10 types 1 and 2 are disposal into nonproductive/productive zones, respectively.
    x = x.loc[x.type_uic.isin(["1", "2"]) & x.vol_gas.eq(0)].copy()
    x = x.rename(columns={"uic_no": "well_id", "formatted_date": "source_date", "vol_liq": "volume_bbl"})
    return "Texas_Southern_Delaware", eq, x, "M"


def save_region(name, eq, reports, frequency, min_training_events=0):
    report = reports[["well_id", "source_date", "volume_bbl"]].copy()
    report.source_date = pd.to_datetime(report.source_date).dt.normalize()
    if report.well_id.isna().any() or report.source_date.isna().any():
        raise ValueError(f"Missing well identifier or period: {name}")
    report.volume_bbl = pd.to_numeric(report.volume_bbl, errors="raise")
    invalid = ~np.isfinite(report.volume_bbl) | report.volume_bbl.lt(0)
    report.loc[invalid, "volume_bbl"] = np.nan
    grouped = report.groupby(["source_date", "well_id"]).volume_bbl
    conflicts = grouped.nunique() > 1
    well = grouped.max().mask(conflicts).rename("volume_bbl").reset_index()
    total = well.groupby("source_date").volume_bbl
    inj = pd.DataFrame({"reported_volume_bbl": total.sum(min_count=1), "reporting_wells": total.count(),
                        "listed_wells": total.size()}).reset_index()
    end = inj.source_date + (pd.offsets.MonthBegin(1) if frequency == "M" else pd.Timedelta(days=1))
    start_utc = inj.source_date.dt.tz_localize("America/Chicago").dt.tz_convert("UTC")
    inj["period_end_exclusive"] = end.dt.strftime("%Y-%m-%d")
    # Earliest possible use of a retrospective volume; subsequent filing delays are unknown.
    inj["time_iso"] = end.dt.tz_localize("America/Chicago").dt.tz_convert("UTC")
    minutes = (inj.time_iso - start_utc).dt.total_seconds() / 60
    inj["inj_rate_m3_min"] = inj.reported_volume_bbl * BARREL_M3 / minutes
    period = "%Y-%m" if frequency == "M" else "%Y-%m-%d"
    eq = eq.copy()
    eq["injection_period"] = eq.time_iso.dt.tz_convert("America/Chicago").dt.strftime(period)
    available = inj.loc[inj.reporting_wells.gt(0), "source_date"].dt.strftime(period)
    paired = eq.loc[eq.injection_period.isin(available) & np.isfinite(eq.magnitude)].copy()
    if paired.event_id.isna().any() or paired.event_id.duplicated().any() or paired.time_iso.isna().any():
        raise ValueError(f"Ambiguous event identity or clock: {name}")
    # Some OGS years publish times only to the second. Avoid zero inter-event times.
    tied = paired.time_iso.duplicated(keep=False)
    paired = paired.loc[~tied].sort_values("time_iso").reset_index(drop=True)
    if len(paired) <= 1500:
        raise ValueError(f"Only {len(paired)} unambiguous marked, paired events: {name}")
    def boundary(fraction):
        i = int(len(paired) * fraction)
        t0, t1 = paired.time_iso.iloc[i-1:i+1]
        return t0 + (t1 - t0) / 2
    val, test = boundary(.70), boundary(.85)
    mc = round(float(estimate_mc_max_curvature(paired.loc[paired.time_iso < val, "magnitude"], bin_width=.1, plot=False)[0]), 2)
    scored = paired.loc[paired.magnitude > mc, "time_iso"]
    training_events = int((scored < val).sum())
    if training_events < min_training_events:
        raise ValueError(f"{name} has {training_events} training events above Mc, requires {min_training_events}")
    summary = dict(dataset=name, events=len(paired), with_magnitude=len(paired),
                   spatial_events_before_pairing=len(eq), tied_time_rows_excluded=int(tied.sum()),
                   invalid_volume_rows=int(invalid.sum()), conflicting_well_periods=int(conflicts.sum()),
                   repeat_well_period_rows=len(report)-len(well), injection_rows=len(inj),
                   periods_with_flow=int(inj.reporting_wells.gt(0).sum()),
                   wells_with_valid_reports=int(well.loc[well.volume_bbl.notna(), "well_id"].nunique()),
                   pump_resolution="calendar month" if frequency == "M" else "calendar day",
                   first_event=paired.time_iso.iloc[0].isoformat(), last_event=paired.time_iso.iloc[-1].isoformat(),
                   val_start_ts=val.isoformat(), test_start_ts=test.isoformat(), mc=mc,
                   mc_method="training-prefix MAXC, bin width 0.1, retain M > Mc",
                   n_scored=[int((scored < val).sum()), int(((scored >= val) & (scored < test)).sum()), int((scored >= test).sum())])
    dest = DATA / name / "processed"
    dest.mkdir(parents=True, exist_ok=True)
    paired.to_csv(dest / f"{name}_eq_processed.csv", index=False)
    inj.to_csv(dest / f"{name}_inj_processed.csv", index=False)
    well.to_csv(dest / f"{name}_inj_by_well.csv", index=False)
    write_json(dest / f"{name}_summary.json", summary)
    print(name, len(paired), len(inj), summary["wells_with_valid_reports"], flush=True)


if __name__ == "__main__":
    for item in json.loads((RAW.parent / "sources.json").read_text())["files"]:
        with (RAW / item["file"]).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"Input checksum mismatch: {item['file']}")
    for dataset in [kansas(), delaware()]:
        save_region(*dataset)
    for dataset in oklahoma():
        save_region(*dataset)
