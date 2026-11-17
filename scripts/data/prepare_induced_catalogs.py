"""Rebuild five observational catalogs from their pinned raw files. Requires pandas and numpy."""
import argparse
import gzip
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
GPM_TO_M3_MIN = 0.003785411784


def newberry(raw):
    eq = pd.read_csv(raw / "cataloguedelimited-529f6605f226d.dat", sep="\t")
    events = pd.DataFrame({
        "event_id": "newberry:row:" + (eq.index + 2).astype(str),
        "time_iso": pd.to_datetime(eq.Date + " " + eq.Time, format="%m/%d/%Y %H:%M:%S", utc=True),
        "magnitude": eq.Mag, "magnitude_type": eq.Magt.replace("_", pd.NA),
        "latitude": eq.Lat, "longitude": eq.Lon, "depth_source": eq.Depth,
        "easting_utm10_m": eq.easting, "northing_utm10_m": eq.northing,
    })
    pump = pd.read_csv(raw / "WHP_Flow.csv")
    local = pd.to_datetime(pump.Date + " " + pump.Time, format="%m/%d/%Y %H:%M:%S")
    utc = local.dt.tz_localize("America/Los_Angeles", ambiguous="infer").dt.tz_convert("UTC")
    if utc.duplicated().any() or not utc.is_monotonic_increasing:
        raise ValueError("Ambiguous Newberry pump chronology")
    flow = pd.to_numeric(pump.Downhole_Flow.replace({"LINE": None, "DRAINED": None}), errors="raise")
    inj = pd.DataFrame({"time_iso": utc, "inj_rate_m3_min": flow * GPM_TO_M3_MIN,
                        "wellhead_pressure_psi": pump.WHP, "downhole_flow_source": pump.Downhole_Flow})
    return events, inj, {"raw_events": len(eq), "missing_downhole_flow": int(flow.isna().sum()),
                         "pump_gaps_over_1min": int((utc.diff() > pd.Timedelta(minutes=1)).sum())}


def zip_csv(path):
    with ZipFile(path) as archive:
        member, = [n for n in archive.namelist() if n.endswith(".csv")]
        with archive.open(member) as f:
            return pd.read_csv(f)


def paradox(raw):
    source = zip_csv(raw / "earthquake.zip")
    # USBR uses a wellhead datum of 1540 m MSL and a shallow-event limit of 8.5 km.
    depth_wellhead = (1540.0 - source["Elevation_(m)"]) / 1000
    eq = source.loc[(source.Quality == "a") & (source.Year >= 1991) & (depth_wellhead < 8.5)].copy()
    clock = eq[["Year", "Month", "Day", "Hour", "Minute"]].rename(columns=str.lower)
    time = pd.to_datetime(clock, utc=True) + pd.to_timedelta(eq.Second, unit="s")
    events = pd.DataFrame({
        "event_id": eq.Event_ID, "time_iso": time, "magnitude": eq.Magnitude,
        "magnitude_type": eq.Magnitude_type, "magnitude_source": eq.Magnitude_source,
        "latitude": eq["Latitude_(deg)"], "longitude": eq["Longitude_(deg)"],
        "elevation_m_msl": eq["Elevation_(m)"], "depth_km_below_wellhead": depth_wellhead.loc[eq.index],
        "location_quality": eq.Quality,
        "station_count": eq.Nstations, "rms_seconds": eq["RMS_residual_(s)"],
    })
    pump = zip_csv(raw / "injection.zip")
    dates = pd.to_datetime(pump.Date, format="%m/%d/%y")
    if dates.duplicated().any():
        raise ValueError("Duplicate Paradox Valley daily reading")
    inj = pd.DataFrame({"source_date": dates.dt.strftime("%Y-%m-%d"),
                        "mean_inj_rate_m3_min": pump["Injection Rate (gpm)"] * GPM_TO_M3_MIN,
                        "mean_surface_pressure_psi": pump["Surface Pressure (psi)"]})
    return events, inj, {"raw_events": len(source), "selection": "USBR a-quality, injection era, <8.5 km below wellhead",
                         "missing_daily_flow": int(inj.mean_inj_rate_m3_min.isna().sum())}


def sheet_rows(book, number, strings, columns):
    # Published figure-source workbook. Read the cells with values, ignoring the large styled blank range.
    with book.open(f"xl/worksheets/sheet{number}.xml") as f:
        for _, row in ET.iterparse(f, events=["end"]):
            if not row.tag.endswith("}row"):
                continue
            values = {}
            for cell in row.findall("{*}c"):
                col = cell.get("r").rstrip("0123456789")
                value = cell.findtext("{*}v")
                if col in columns and value is not None:
                    values[col] = strings[int(value)] if cell.get("t") == "s" else value
            if values:
                yield int(row.get("r")), values
            row.clear()


def pohang(raw):
    with ZipFile(raw / "41467_2020_16408_MOESM4_ESM.xlsx") as book:
        strings = ["".join(c.itertext()) for c in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        rows = [(i, d) for i, d in sheet_rows(book, 1, strings, {"M", "N", "O"}) if i > 2 and "M" in d]
        events = pd.DataFrame({
            "event_id": [f"fig2a:row:{i}" for i, _ in rows],
            "time_iso": pd.to_datetime([d["M"] for _, d in rows], format="%m/%d/%Y %H:%M:%S.%f", utc=True),
            "magnitude": [float(d["N"]) for _, d in rows], "magnitude_type": "ML",
            "Mw": [float(d["O"]) if d.get("O", "-") != "-" else np.nan for _, d in rows],
        })
        campaigns = []
        for number, well in enumerate(["PX-2", "PX-1", "PX-2", "PX-1", "PX-2"], start=2):
            readings = [d for i, d in sheet_rows(book, number, strings, {"A", "B", "C"}) if i > 1 and "A" in d]
            frame = pd.DataFrame(readings).astype(float)
            kst = pd.to_datetime(frame.A, unit="D", origin="1899-12-30").dt.round("ms")
            campaigns.append(pd.DataFrame({"time_iso": kst.dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC"),
                                           "well": well, "stimulation": number - 1,
                                           "inj_rate_m3_min": frame.B * .06, "wellhead_pressure_MPa": frame.C}))
    inj = pd.concat(campaigns, ignore_index=True)
    return events, inj, {"raw_events": len(events), "with_Mw": int(events.Mw.notna().sum()),
                         "source": "Figure 2a selected earthquakes; Figure 2b-f reported flow rates"}


def aspo(raw):
    source = pd.read_csv(raw / "aspo_catalog_niemz2020.csv", sep=";", dtype={"name": str})
    unique = source.drop_duplicates()
    conflicts = unique.name.duplicated(keep=False)
    eq = unique.loc[~conflicts].copy()
    time = pd.to_datetime(eq.name, format="%Y%m%d%H%M%S.%f", utc=True)
    other = pd.to_datetime(eq.origin_time, format="%Y-%m-%d_%H:%M:%S.%f", utc=True)
    if not ((other.dt.second == other.dt.day) & (time.dt.floor("min") == other.dt.floor("min"))
            & (time.dt.microsecond == other.dt.microsecond)).all():
        raise ValueError("Unexpected change in GFZ time encoding; see continous.readme")
    events = pd.DataFrame({"event_id": eq.name, "time_iso": time, "source_origin_time": eq.origin_time,
                           "magnitude": eq.MAE, "magnitude_type": "MAE", "x_aspo96_m": eq.x,
                           "y_aspo96_m": eq.y, "depth_aspo96_m": eq.depth})
    events["hydraulic_record"] = pd.Series(index=events.index, dtype="string")
    bins = []
    with tarfile.open(raw / "hydraulic_data.tar.gz") as archive:
        for item in sorted(archive.getmembers(), key=lambda m: m.name):
            if not item.isfile() or "/hf/" not in item.name or not item.name.endswith(".csv"):
                continue
            with archive.extractfile(item) as f:
                d = pd.read_csv(f, sep=";", usecols=["unix_time[sec]", "flow[l/min]", "interval_pressure[MPa]"])
            ms = np.rint(d["unix_time[sec]"] * 1000).astype("int64")
            idx = pd.DatetimeIndex(pd.to_datetime(ms, unit="ms", utc=True))
            if not idx.is_monotonic_increasing or idx.has_duplicates:
                raise ValueError(f"Unordered hydraulic record: {item.name}")
            record = Path(item.name).stem
            inside = events.time_iso.between(idx[0], idx[-1])
            if events.loc[inside, "hydraulic_record"].notna().any():
                raise ValueError("Overlapping Äspö records")
            events.loc[inside, "hydraulic_record"] = record
            groups = d.set_index(idx).resample("1s", label="right", closed="right")
            out = groups.agg(inj_rate_m3_min=("flow[l/min]", "mean"),
                             interval_pressure_MPa=("interval_pressure[MPa]", "mean"),
                             observations=("unix_time[sec]", "size"))
            out.inj_rate_m3_min /= 1000
            bins.append(out.loc[out.observations > 0].rename_axis("time_iso").reset_index().assign(hydraulic_record=record))
    paired = events.loc[events.hydraulic_record.notna()].copy()
    return paired, pd.concat(bins, ignore_index=True), {
        "raw_events": len(source), "exact_repeat_rows_removed": len(source) - len(unique),
        "conflicting_id_rows_excluded": int(conflicts.sum()), "outside_pump_span": len(events) - len(paired),
        "seconds_field": "UTC event name per GFZ format; source_origin_time has day in seconds position",
        "with_MAE": int(paired.magnitude.notna().sum()), "absolute_pump_timing_precision_seconds": 1}


def knmi(raw):
    with gzip.open(raw / "induced_1986_2025.xml.gz") as f:
        source = ET.parse(f).findall(".//{*}event")
    rows = []
    for e in source:
        if e.findtext("{*}type") != "induced or triggered event":
            raise ValueError("Unexpected KNMI event type")
        def preferred(tag, id_tag):
            return next(n for n in e.findall("{*}" + tag) if n.get("publicID") == e.findtext("{*}" + id_tag))
        origin = preferred("origin", "preferredOriginID")
        mag = preferred("magnitude", "preferredMagnitudeID")
        status = origin.findtext("{*}evaluationStatus")
        if origin.findtext("{*}evaluationMode") != "manual" or status not in {"final", "reviewed", "confirmed"}:
            continue
        def value(node, field):
            return node.findtext("{*}" + field + "/{*}value")
        rows.append(dict(event_id=e.get("publicID"), time_iso=value(origin, "time"), magnitude=float(value(mag, "mag")),
                         magnitude_type=mag.findtext("{*}type"), latitude=float(value(origin, "latitude")),
                         longitude=float(value(origin, "longitude")), depth_km=float(value(origin, "depth")) / 1000,
                         origin_status=status, origin_id=origin.get("publicID"), magnitude_id=mag.get("publicID")))
    events = pd.DataFrame(rows)
    events.time_iso = pd.to_datetime(events.time_iso, format="mixed", utc=True)
    return events, None, {"raw_events": len(source), "excluded_origin_quality": len(source)-len(events)}


BUILDERS = {"Newberry_EGS2012": newberry, "Paradox_Valley": paradox, "Pohang_EGS": pohang,
            "Aspo_HRL2015": aspo, "KNMI_Induced": knmi}


def build(name, data_dir):
    site = data_dir / name
    raw = site / "raw"
    for entry in json.loads((site / "sources.json").read_text(encoding="utf-8"))["files"]:
        with (raw / entry["file"]).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f'Source checksum mismatch: {entry["file"]}')
    eq, injection, audit = BUILDERS[name](raw)
    eq = eq.sort_values(["time_iso", "event_id"], kind="stable")
    if eq.empty or eq.event_id.isna().any() or eq.event_id.duplicated().any() or eq.time_iso.isna().any():
        raise ValueError(f"Invalid event identities/timestamps: {name}")
    out = site / "processed"
    out.mkdir(exist_ok=True)
    eq.to_csv(out / f"{name}_eq_processed.csv", index=False, lineterminator="\n")
    if injection is not None:
        clock = "time_iso" if "time_iso" in injection else "source_date"
        injection = injection.sort_values(clock, kind="stable")
        if injection.empty or injection[clock].isna().any() or injection[clock].duplicated().any():
            raise ValueError(f"Invalid injection timestamps: {name}")
        injection.to_csv(out / f"{name}_inj_processed.csv", index=False, lineterminator="\n")
    audit.update(dataset=name, events=len(eq), with_magnitude=int(eq.magnitude.notna().sum()),
                 tied_time_rows=int(eq.time_iso.duplicated(keep=False).sum()),
                 first_event=eq.time_iso.min().isoformat(), last_event=eq.time_iso.max().isoformat(),
                 injection_rows=0 if injection is None else len(injection))
    (out / f"{name}_summary.json").write_bytes((json.dumps(audit, indent=2) + "\n").encode())
    print(name, audit["events"], audit["with_magnitude"], audit["injection_rows"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--dataset", choices=BUILDERS)
    args = parser.parse_args()
    for name in ([args.dataset] if args.dataset else BUILDERS):
        build(name, args.data_dir)
