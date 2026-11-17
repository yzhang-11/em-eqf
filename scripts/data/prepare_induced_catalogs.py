"""Rebuild induced-seismicity catalogs from their pinned raw files. Requires pandas and numpy; FORGE2024 also uses python-calamine."""
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
MIN_MARKED_EVENTS = 500
BBL_M3 = 0.158987294928
GPM_TO_M3_MIN = 0.003785411784


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
    # Read only populated cells from the published XLSX.
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
                raise ValueError("Overlapping 脛sp枚 records")
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


def tony_creek(raw):
    with ZipFile(raw / "Catalog_JGR.xlsx") as book:
        strings = ["".join(node.itertext()) for node in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        rows = sheet_rows(book, 1, strings, set("ABCDEFGHIJ"))
        _, header = next(rows)
        fields = ["Number", "Month", "Day", "Hour", "Minute", "Second", "Easting (m)",
                  "Northing (m)", "Depth (m)", "Magnitude Mw"]
        columns = list("ABCDEFGHIJ")
        if [header[c] for c in columns] != fields:
            raise ValueError("Unexpected ToC2ME workbook header")
        eq = pd.DataFrame([row for _, row in rows], columns=columns).astype(float)
    clock = eq[list("BCDEF")].set_axis(["month", "day", "hour", "minute", "second"], axis=1)
    events = pd.DataFrame({
        "event_id": eq.A.astype("int64"), "time_iso": pd.to_datetime(clock.assign(year=2016), utc=True),
        "magnitude": eq.J, "magnitude_type": "Mw", "easting_local_m": eq.G,
        "northing_local_m": eq.H, "depth_source_m": eq.I,
    })
    return events, None, {"raw_events": len(eq), "time_resolution_seconds": 1}


def kiskatinaw(raw):
    eq = pd.read_csv(raw / "EarthquakeCatalog_KiskatinawArea_Rothetal_Dec2020.csv")
    if not eq.event_type.eq("induced event").all():
        raise ValueError("Unexpected Kiskatinaw event type")
    events = eq.rename(columns={"ID": "event_id", "origin": "time_iso", "mag_type": "magnitude_type",
                                "depth": "depth_km"}).drop(columns=["date", "time"])
    events.time_iso = pd.to_datetime(events.time_iso, utc=True, format="ISO8601")
    return events, None, {"raw_events": len(eq)}


def forge2024(raw):
    # Calfrac instrument clocks are fixed MST (UTC-07), including during April.
    events = pd.read_csv(raw / "microseismic_catalog_Utah_FORGE_2024stim_niemzetal2024.csv", dtype={"ID": str})
    events = events.rename(columns={"ID": "event_id", "time(UTC)": "time_iso", "lat": "latitude",
                                    "lon": "longitude", "depth[m_local]": "depth_local_m",
                                    "east_shift[m_local]": "easting_local_m", "north_shift[m_local]": "northing_local_m"})
    events.time_iso = pd.to_datetime(events.time_iso, utc=True)
    events["magnitude_type"] = "ML_calibrated"
    events["hydraulic_record"] = pd.Series(index=events.index, dtype="string")
    frames = []
    for well in ("16A", "16B"):
        with ZipFile(raw / f"{well} Frac Data.zip") as archive:
            for member in sorted(n for n in archive.namelist() if n.endswith(".xlsx")):
                stage = member.split("_Stage ")[1].split("_")[0]
                record = f"{well}-{stage}"
                with archive.open(member) as stream:
                    x = pd.read_excel(stream, header=None, usecols="A:G", engine="calamine")
                if list(x.iloc[3, [0, 1, 2, 6]]) != ["Date", "Day", "Pressure", "Rate"] or x.iloc[4, 6] != "(bpm)":
                    raise ValueError(f"Unexpected Calfrac headers: {member}")
                x = x.iloc[5:].dropna(how="all")
                time = pd.to_datetime(x[0]).dt.normalize() + pd.to_timedelta(x[1].astype(str))
                utc = time.dt.tz_localize("Etc/GMT+7").dt.tz_convert("UTC")
                rate = pd.to_numeric(x[6], errors="raise")
                inj = pd.DataFrame({"time_iso": utc, "well": well, "hydraulic_record": record,
                                    "clean_rate_bbl_min": rate, "inj_rate_m3_min": rate * BBL_M3,
                                    "treating_pressure_psi": pd.to_numeric(x[2], errors="raise")})
                if not utc.is_monotonic_increasing or utc.duplicated().any():
                    raise ValueError(f"Unordered Calfrac records: {member}")
                # Missing samples interrupt a sequence. Only bracketed events have known coverage.
                measured = inj.loc[np.isfinite(rate) & (rate >= 0)].copy()
                parts = measured.time_iso.diff().ne(pd.Timedelta(seconds=1)).cumsum()
                for part, group in measured.groupby(parts):
                    keep = events.time_iso.between(group.time_iso.iloc[0], group.time_iso.iloc[-1])
                    if events.loc[keep, "hydraulic_record"].notna().any():
                        raise ValueError("Overlapping Calfrac observation windows")
                    label = f"{record}.{part}"
                    events.loc[keep, "hydraulic_record"] = label
                    frames.append(group.assign(hydraulic_record=label))
    paired = events.hydraulic_record.notna()
    return events, pd.concat(frames, ignore_index=True), {
        "raw_events": len(events), "paired_events": int(paired.sum()), "outside_measured_windows": int((~paired).sum()),
        "site_group": "Utah FORGE", "clock": "Calfrac fixed MST (UTC-07)",
        "rate_source": "original one-second Calfrac clean fluid rate (bbl/min)",
        "measured_windows": len(frames), "windows_with_events": events.hydraulic_record.nunique(),
        "wells": sorted({f.well.iloc[0] for f in frames})}


def lacq(raw):
    columns = ["latitude", "longitude", "depth_km", "magnitude", "year", "month", "day", "hour",
               "minute", "second", "rms_seconds", "n_phases", "horizontal_uncertainty_km",
               "vertical_uncertainty_km", "azimuthal_gap_deg", "pgv_max_mm_s", "mw_from_ml"]
    events = pd.read_csv(raw / "CATALOG_LACQ.txt", sep=r"\s+", comment="#", names=columns)
    clock = events[["year", "month", "day", "hour", "minute"]].astype(int)
    time = pd.to_datetime(clock, utc=True) + pd.to_timedelta(events.second, unit="s")
    if time.duplicated().any():
        raise ValueError("Ambiguous Lacq origin time used as event identity")
    events = events.drop(columns=[*clock.columns, "second"])
    events.insert(0, "event_id", "lacq:" + time.dt.strftime("%Y%m%dT%H%M%S.%fZ"))
    events.insert(1, "time_iso", time)
    events["magnitude_type"] = "ML"
    return events, None, {"raw_events": len(events), "selection": "author-constrained CATALOG_LACQ"}


BUILDERS = {"Paradox_Valley": paradox, "Aspo_HRL2015": aspo, "KNMI_Induced": knmi,
            "Tony_Creek": tony_creek, "Kiskatinaw": kiskatinaw, "FORGE2024": forge2024,
            "Lacq_SISLACQ2021": lacq}


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
    if np.isinf(eq.magnitude).any() or eq.magnitude.notna().sum() < MIN_MARKED_EVENTS:
        raise ValueError(f"Invalid magnitudes or fewer than {MIN_MARKED_EVENTS} marked events: {name}")
    out = site / "processed"
    out.mkdir(exist_ok=True)
    if name == "FORGE2024":
        eq.loc[eq.hydraulic_record.notna()].to_csv(out / f"{name}_eq_paired_processed.csv", index=False, lineterminator="\n")
        eq = eq.drop(columns="hydraulic_record")
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
