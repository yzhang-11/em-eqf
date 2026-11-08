"""Verify and normalize three versioned source archives for em_eqf."""
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

STORE = Path("/root/autodl-fs/em_eqf_catalogs/source_expansion_20261106")
RAW = STORE / "raw"
OUT = Path("/root/autodl-tmp/em_eqf/experiments/revision/catalog_source_expansion_20261106")
FIELDS = ["latitude", "longitude", "depth_km", "magnitude"]
POLICIES = {
    "Taiwan_CWA": "Historical manual quality-controlled CWA archive. ML>=3, source review code F, at least 8 stations, gap<=180 deg, RMS<=1 s, ERH<=2 km, ERZ<=5 km.",
    "Taiwan_Wu2008": "Author relocated event solutions. Published magnitude>=2.5, quality A/B, free depth, at least 8 stations, gap<=180 deg, RMS<=1 s, ERH<=2 km, ERZ<=5 km.",
    "Turkiye_EAFZ": "Author CASP/NLLoc research solutions. Recalibrated ML>=2.5, at least 12 P+S phases, gap<=180 deg, ERH<=2 km, ERZ<=5 km. Phase count is distinct from station count.",
}


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def utc(values):
    return pd.to_datetime(values, utc=True, format="mixed", errors="raise").as_unit("ns")


def common_quality(frame):
    return (
        frame.gap_deg.between(0, 180)
        & frame.erh_km.between(0, 2)
        & frame.erz_km.between(0, 5)
    )


def station_quality(frame):
    return common_quality(frame) & (frame.station_count >= 8) & frame.rms_s.between(0, 1)


def cwa(path):
    columns = {
        "OriginTime": "source_time", "EpicenterLatitude": "latitude",
        "EpicenterLongitude": "longitude", "FocalDepth": "depth_km",
        "LocalMagnitude": "magnitude", "StationNumber": "station_count",
        "PhaseNumber": "phase_count", "MinimumEpicenterDistance": "dmin_km",
        "gap": "gap_deg", "rms": "rms_s", "erh": "erh_km", "erz": "erz_km",
        "Quality": "source_quality", "ReviewStatus": "source_review_status",
    }
    annual = []
    frames = []
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        members = sorted(n for n in archive.namelist() if re.fullmatch(r"CWA-EQ-Catalog-\d{4}\.xml", n))
        assert [int(n[-8:-4]) for n in members] == list(range(1990, 2026))
        for member in members:
            data = ET.fromstring(archive.read(member))
            for node in data.iter():
                node.tag = node.tag.rsplit("}", 1)[-1]
            catalog = data.find("./Dataset/Catalog")
            info = catalog.find("CatalogInfo")
            elements = catalog.findall("EarthquakeInfo")
            year = int(member[-8:-4])
            assert len(elements) == int(info.findtext("RecordNumber")), member
            rows = [{out: node.findtext(src) for src, out in columns.items()} for node in elements]
            frame = pd.DataFrame(rows)
            assert frame.source_time.str.endswith("+08:00").all(), member
            frame["time"] = utc(frame.source_time.tolist())
            start, end = utc([info.findtext("StartTime"), info.findtext("EndTime")])
            assert start == pd.Timestamp(f"{year}-01-01", tz="UTC")
            assert end == pd.Timestamp(f"{year+1}-01-01", tz="UTC")
            assert ((frame.time >= start) & (frame.time < end)).all(), member
            frame["event_id"] = [f"cwa:{year}:{i}" for i in range(1, len(frame) + 1)]
            frame["source_member"] = member
            frame["source_year"] = year
            frames.append(frame)
            annual.append({"year": year, "records": len(frame), "declared_range": info.findtext("MagnitudeRange")})
    frame = pd.concat(frames, ignore_index=True)
    for key in FIELDS + ["station_count", "phase_count", "dmin_km", "gap_deg", "rms_s", "erh_km", "erz_km"]:
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
    frame["mag_type"] = "ML"
    keep = (
        frame.source_review_status.eq("F") & station_quality(frame)
        & frame.magnitude.ge(3) & frame.source_year.ge(1991)
        & frame.latitude.between(20, 27) & frame.longitude.between(118, 124)
        & frame.depth_km.between(0, 700)
    )
    return frame, keep, {"archive_annual": annual, "clock": "Explicit +08:00 per XML event, converted to UTC."}


def wu(path):
    columns = [
        "source_date", "source_hour_minute", "source_seconds",
        *FIELDS, "station_count", "dmin_km", "gap_deg", "rms_s",
        "erh_km", "erz_km", "depth_control", "phase_count", "source_quality",
    ]
    expected = [4634, 4322, 5242, 16329, 14018, 15194, 14630, 14254, 46475,
                22913, 15491, 27030, 24646, 20841, 21192]
    frames = []
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {f"{y}.txt" for y in range(1991, 2006)}
        for year, count in zip(range(1991, 2006), expected):
            member = f"{year}.txt"
            rows = [s.split() for s in archive.read(member).decode("ascii").splitlines() if s.strip()]
            assert len(rows) == count and all(len(r) == len(columns) for r in rows), member
            frame = pd.DataFrame(rows, columns=columns)
            frame["event_id"] = [f"wu2008:{year}:{i}" for i in range(1, len(rows) + 1)]
            frame["source_member"] = member
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    base = utc((frame.source_date + " " + frame.source_hour_minute).tolist())
    # Preserve hundredths exactly, including any source seconds field at a rollover.
    assert frame.source_seconds.str.fullmatch(r"\d{1,2}\.\d{2}").all()
    centiseconds = frame.source_seconds.str.replace(".", "", regex=False).astype("int64")
    frame["time"] = base + pd.to_timedelta(centiseconds.to_numpy() * 10, unit="ms")
    for key in FIELDS + ["station_count", "phase_count", "dmin_km", "gap_deg", "rms_s", "erh_km", "erz_km"]:
        frame[key] = pd.to_numeric(frame[key], errors="raise")
    assert ((frame.time >= "1991-01-01") & (frame.time < "2006-01-01")).all()
    chi = frame[frame.time.eq(pd.Timestamp("1999-09-20T17:47:15.690Z"))]
    hua = frame[frame.time.eq(pd.Timestamp("2002-03-31T06:52:50.370Z"))]
    assert len(chi) == len(hua) == 1 and chi.magnitude.iloc[0] >= 7 and hua.magnitude.iloc[0] >= 6.7
    frame["mag_type"] = "published_magnitude"
    keep = (
        frame.depth_control.eq("F") & frame.source_quality.isin(["A", "B"])
        & station_quality(frame) & frame.magnitude.ge(2.5)
        & frame.latitude.between(20, 27) & frame.longitude.between(118, 124)
        & frame.depth_km.between(0, 700)
    )
    return frame, keep, {
        "clock": "UTC inferred from two independent mainshock anchors and checked against overlapping CWA UTC times.",
        "anchors": [
            ["1999-09-20T17:47:15.690Z", "https://www.strongmotioncenter.org/vdc/scripts/event.plx?evt=629"],
            ["2002-03-31T06:52:50.370Z", "https://earthquake.usgs.gov/earthquakes/eventpage/usp000b1e2"],
        ],
    }


def eafz(path):
    metadata = json.loads((RAW / "eafz_zenodo.json").read_text(encoding="utf-8"))
    item = next(item for item in metadata["files"] if item["key"] == "events_catalog.csv")
    assert metadata["id"] == 15910095
    assert item["checksum"] == "md5:" + digest(path, "md5") and item["size"] == path.stat().st_size
    frame = pd.read_csv(path, dtype={"Id_AFAD": "string", "Sec": "string"})
    assert len(frame) == 9442 and frame.Id_AFAD.is_unique and frame.Id_AFAD.notna().all()
    parts = frame[["Year", "Month", "Day", "Hour", "Min"]].rename(columns={"Min": "minute"})
    base = pd.to_datetime(parts, utc=True)
    assert frame.Sec.str.fullmatch(r"\d{1,2}(\.\d{1,2})?").all()
    seconds = frame.Sec.str.split(".", n=1, expand=True)
    centiseconds = seconds[0].astype("int64") * 100 + seconds[1].fillna("").str.pad(2, side="right", fillchar="0").astype("int64")
    assert centiseconds.between(0, 5999).all()
    frame["time"] = base + pd.to_timedelta(centiseconds * 10, unit="ms")
    anchor = frame.loc[frame.Id_AFAD.eq("565481"), "time"].iloc[0]
    assert anchor == pd.Timestamp("2023-03-20T15:40:34.23Z")
    assert ((frame.time >= "2019-01-01") & (frame.time < "2024-03-01")).all()
    frame = frame.rename(columns={
        "Ev_Lat": "latitude", "Ev_Lon": "longitude", "Ev_Depth": "depth_km",
        "Ml": "magnitude", "Erh": "erh_km", "Erz": "erz_km", "Gap": "gap_deg",
        "Ntot": "phase_count", "Rms": "source_rms",
    })
    assert frame.phase_count.eq(frame.Np + frame.Ns).all()
    frame["event_id"] = "afad:" + frame.Id_AFAD
    frame["mag_type"] = "ML"
    keep = common_quality(frame) & frame.phase_count.ge(12) & frame.magnitude.ge(2.5)
    return frame, keep, {
        "clock": "UTC cross-checked against the same relocated event in the paper, Fig. 12 (15:40:34 UTC).",
        "zenodo_version_doi": metadata["doi"], "source_md5": item["checksum"],
        "selection": "Author publication applies waveform-quality and station criteria and an ML upper limit near 5.5. Sample is suitable for a research protocol with this selection, which excludes the 2023 mainshocks.",
    }


def main():
    reports = OUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    sources = [
        ("Taiwan_CWA", "cwa_historical.zip", cwa,
         "https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Earthquake/E-A0073-002.zip",
         "https://www.cwa.gov.tw/Data/data_catalog/5-1-3.pdf"),
        ("Taiwan_Wu2008", "taiwan_wu_1991_2005.zip", wu,
         "https://seismosoc.org/Publications/BSSA_html/bssa_98-3/2007166-esupp/catalogs_1991-2005.zip",
         "https://seismosoc.org/Publications/BSSA_html/bssa_98-3/2007166-esupp/"),
        ("Turkiye_EAFZ", "eafz_events_catalog.csv", eafz,
         "https://zenodo.org/api/records/15910095/files/events_catalog.csv/content",
         "https://essd.copernicus.org/articles/17/3089/2025/"),
    ]
    summary = {}
    for name, filename, reader, url, documentation in sources:
        path = RAW / filename
        frame, keep, notes = reader(path)
        frame["provider"] = name
        frame["absolute_time_ready"] = True
        frame["quality_screen_pass"] = keep & np.isfinite(frame[FIELDS]).all(axis=1)
        assert frame.event_id.is_unique
        selected = frame[frame.quality_screen_pass].sort_values(["time", "event_id"], kind="stable")
        aliases = []
        if name == "Turkiye_EAFZ":
            physical_key = ["time", *FIELDS]
            repeated = selected.loc[selected.duplicated(physical_key, keep=False)]
            for _, group in repeated.groupby(physical_key, sort=False):
                retained = group.sort_values("Id_AFAD", key=lambda ids: ids.astype("int64"))
                representative = retained.event_id.iloc[0]
                aliases.extend({"representative_id": representative, "alias_id": alias}
                               for alias in retained.event_id.iloc[1:])
            selected = selected.loc[~selected.event_id.isin([row["alias_id"] for row in aliases])]
            pd.DataFrame(aliases, columns=["representative_id", "alias_id"]).to_csv(
                reports / "eafz_identical_physical_records.csv", index=False,
            )
        destination = STORE / name / "processed"
        destination.mkdir(parents=True, exist_ok=True)
        output = {}
        for key, table in [("normalized_all", frame), ("quality_screened", selected)]:
            target = destination / (key + ".parquet")
            temporary = target.with_suffix(".parquet.part")
            table.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(target)
            output[key] = {"path": str(target), "sha256": digest(target), "records": len(table)}
        summary[name] = {
            "raw": {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path), "url": url},
            "documentation": documentation, "quality_policy": POLICIES[name], "artifacts": output,
            "by_year": {str(y): int(n) for y, n in selected.time.dt.year.value_counts().sort_index().items()},
            "source_review_codes": frame.source_review_status.value_counts(dropna=False).to_dict()
            if "source_review_status" in frame else None,
            "source_physical_duplicate_rows": int(frame.duplicated(["time", *FIELDS]).sum()),
            "identical_physical_aliases_excluded_from_standard": len(aliases),
            **notes,
        }
        print(name, len(frame), "->", len(selected), flush=True)
    save_json(reports / "sources.json", summary)


if __name__ == "__main__":
    main()
