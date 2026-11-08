"""Build new source snapshots for corroborated NCEDC events and reviewed ComCat Montana."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data import download_more_catalogs as regional
from scripts.data import expand_catalogs_20261105 as expansion
from src.catalogs.external_catalog import file_sha256

STORE = Path("/root/autodl-fs/em_eqf_catalogs/remediation_20261106")
REPORTS = ROOT / "experiments/revision/catalog_remediation_20261106/reports"
AUDIT = ROOT / "experiments/revision/catalog_integrity_20261106/reports"
MONTANA = "USGS_MontanaComCat"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_parquet(path, frame):
    temporary = path.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    return {"path": str(path), "sha256": file_sha256(path), "records": len(frame)}


def ncedc():
    previous = ROOT / "experiments/revision/catalog_expansion_20261105/reports/NCEDC_audit.json"
    manifest = json.loads(previous.read_text())
    old, strict = manifest["outputs"]
    source_check = json.loads((AUDIT / "source_gap_checks.json").read_text())["NCEDC"]
    raw_file = Path(source_check["cloud_path"])
    raw_info, = (item for item in manifest["raw_manifest"] if item["url"] == source_check["cloud_url"])
    assert file_sha256(raw_file) == raw_info["sha256"]
    for item in (old, strict):
        assert file_sha256(item["path"]) == item["sha256"]

    raw = expansion.load_fdsn([(raw_file, raw_info)], "NCEDC")
    fdsn_file = AUDIT / "ncedc_2024_fdsn.txt"
    fdsn = pd.read_csv(fdsn_file, sep="|", dtype=str)
    fdsn.columns = fdsn.columns.str.strip()
    fdsn = fdsn.rename(columns={"#EventID": "event_id", "Time": "official_time",
                               "Latitude": "official_lat", "Longitude": "official_lon",
                               "EventType": "official_type"})
    joined = raw.merge(fdsn[["event_id", "official_time", "official_lat", "official_lon", "official_type"]],
                       on="event_id", validate="one_to_one")
    cloud_time = pd.to_datetime(joined.time, utc=True, format="mixed")
    official_time = pd.to_datetime(joined.official_time, utc=True, format="mixed")
    dt = (cloud_time - official_time).abs().dt.total_seconds()
    lat, lon, official_lat, official_lon = (
        np.radians(pd.to_numeric(joined[key])) for key in
        ("latitude", "longitude", "official_lat", "official_lon"))
    h = np.sin((official_lat-lat)/2)**2 + np.cos(lat)*np.cos(official_lat)*np.sin((official_lon-lon)/2)**2
    distance = 12742*np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    supported = (~joined.event_type.isin(["eq", "earthquake"]) &
                 joined.official_type.str.strip().eq("earthquake") & dt.le(2) & distance.le(15))
    repaired = joined.loc[supported, raw.columns].copy()
    repaired["original_event_type"] = repaired.event_type
    repaired["event_type"] = "earthquake"
    repaired["event_type_basis"] = "NCEDC FDSN type matched to original cloud ID, time and location"
    repaired, _ = expansion.prepare_frame(repaired, "NCEDC")
    error_fields = ["azimuthal_gap", "rms", "horizontal_error_km", "depth_error_km"]
    nonnegative = repaired[error_fields].ge(0).all(axis=1)
    repaired["quality_screen_pass"] &= nonnegative & np.isfinite(repaired.depth_km)

    expected = pd.read_csv(AUDIT / "ncedc_2024_type_recovery_candidates.csv", dtype={"id": str})
    assert set(repaired.event_id) == set(expected.id)
    fixed = repaired[repaired.quality_screen_pass & repaired.magnitude.ge(2.5)]
    assert set(fixed.event_id) == set(expected.loc[expected.passes_existing_numeric_quality, "id"])
    old_all, old_quality = (pd.read_parquet(item["path"]) for item in (old, strict))
    assert not set(repaired.event_id).intersection(old_all.event_id)
    # Existing rows retain the original preferred solutions, including the QC flags.
    normal = pd.concat([old_all, repaired], ignore_index=True).sort_values(["time", "event_id"], kind="stable")
    normal["same_time_as_previous"] = normal.time.duplicated()
    screened = normal[normal.quality_screen_pass & normal.magnitude.ge(2.5)]
    base = old_quality[old_quality.magnitude.ge(2.5)]
    assert len(screened) == len(base) + len(fixed)
    folder = STORE / "NCEDC" / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    result = {"old_source": old, "old_screened_source": strict, "raw_2024": raw_info,
              "fdsn_query": source_check["query_url"], "fdsn_sha256": file_sha256(fdsn_file),
              "corroborated_type_repairs": len(repaired), "new_quality_m25": len(fixed),
              "normalised": write_parquet(folder / "earthquakes.parquet", normal),
              "quality_screened": write_parquet(folder / "quality_screened.parquet", screened)}
    write_json(REPORTS / "ncedc_source.json", result)
    print("NCEDC", len(repaired), len(fixed), len(screened), flush=True)


def montana():
    regional.STORE = regional.shared.STORE = STORE
    regional.OUT = regional.shared.OUT = REPORTS.parent
    cfg = dict(kind="usgs", url="https://earthquake.usgs.gov/fdsnws/event/1/query",
               box=(43., 50., -117., -104.), mc=2., fetch_minmagnitude=1.999,
               start="2014-01-01", end="2026-01-01",
               reference="https://earthquake.usgs.gov/fdsnws/event/1/")
    regional.SOURCES[MONTANA] = cfg
    with ThreadPoolExecutor(max_workers=2) as pool:
        batches = pool.map(lambda bounds: regional.fetch(MONTANA, *bounds), regional.periods(cfg))
        paths = [item for batch in batches for item in batch]
    rows = regional.shared.load(MONTANA, paths)
    result = regional.shared.curate(MONTANA, rows, paths)
    print("Montana", result["input_rows"], result["artifacts"]["quality_screened"]["records"], flush=True)


if __name__ == "__main__":
    REPORTS.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["ncedc"]:
        ncedc()
    elif sys.argv[1:] == ["montana"]:
        montana()
    else:
        raise SystemExit("Specify ncedc or montana")
