"""Rebuild five additional regional catalogs with more than 1,000 training events.

Use the project environment and pinned agency snapshots. Training counts follow
the existing chronological 70/15/15 split and training-only MAXC threshold.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.prepare_water_regions import areas_for, save_region

DATA = ROOT / "data"
SHARED = DATA / "TexNet_Disposal_Inputs"
TEXAS_REGIONS = {
    "Texas_Culberson": ["Culberson"],
    "Texas_Martin": ["Martin"],
    "Texas_Midland_FiveCounties": ["Midland", "Ector", "Andrews", "Howard", "Glasscock"],
    "Texas_Scurry_Fisher_Mitchell": ["Scurry", "Fisher", "Mitchell"],
    "Texas_EagleFord": ["Karnes", "Atascosa", "Wilson", "Frio", "Gonzales",
                        "Live Oak", "McMullen", "DeWitt", "La Salle"],
}


def check_sources(folder):
    manifest = json.loads((folder / "sources.json").read_text())
    for item in manifest["files"]:
        with (folder / "raw" / item["file"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"Input checksum mismatch: {item['file']}")


def texas():
    raw = SHARED / "raw"
    snapshots = sorted(raw.glob("catalog_[0-9][0-9][0-9].json"))
    events = pd.DataFrame([feature["attributes"] for path in snapshots
                           for feature in json.loads(path.read_text())["features"]])
    expected = json.loads((raw / "catalog_ids.json").read_text())["objectIds"]
    if set(events.EarthquakeId) != set(expected) or events.EventId.duplicated().any():
        raise ValueError("Incomplete or duplicated TexNet snapshot")
    if not (events.EvaluationMode.eq("manual") & events.EvaluationStatus.eq("final")
            & events.EventType.eq("earthquake")).all():
        raise ValueError("Unexpected TexNet review status or event type")
    events = events.rename(columns={
        "EventId": "source_id", "Magnitude": "magnitude", "MagType": "magnitude_type",
        "Latitude": "latitude", "Longitude": "longitude", "Depth": "depth_km",
        "LatitudeError": "latitude_error", "LongitudeError": "longitude_error",
        "DepthUncertainty": "depth_uncertainty", "PhaseCount": "phase_count",
        "EvaluationMode": "evaluation_mode", "EvaluationStatus": "evaluation_status",
        "RMS": "rms_seconds", "CountyName": "reported_county",
    })
    events["event_id"] = "texnet:" + events.source_id.astype(str)
    # Preserve half-millisecond split midpoints when applying the shared splitter.
    events["time_iso"] = pd.to_datetime(events.Event_Date, unit="ms", utc=True).astype("datetime64[ns, UTC]")
    boundaries = json.loads((raw / "tx_counties.json").read_text())
    names = [county for group in TEXAS_REGIONS.values() for county in group]
    events["county"] = areas_for(events.longitude, events.latitude, boundaries, names)
    locations = pd.DataFrame([row for path in sorted(raw.glob("uic_*.json"))
                              for row in json.loads(path.read_text())])
    for column in ("latitude_nad83", "longitude_nad83"):
        locations[column] = pd.to_numeric(locations[column], errors="coerce")
    locations["county"] = areas_for(locations.longitude_nad83, locations.latitude_nad83,
                                   boundaries, names)
    locations = locations.loc[locations.uic_type_injection.isin(["1", "2"])].copy()
    if locations.groupby("uic_number").api_no.nunique().gt(1).any():
        raise ValueError("A UIC permit maps to multiple APIs")
    for name, counties in TEXAS_REGIONS.items():
        wells = locations.loc[locations.county.isin(counties)]
        suffix = name.removeprefix("Texas_")
        reports = pd.DataFrame([row for path in sorted(raw.glob(f"h10_{suffix}_*.json"))
                                for row in json.loads(path.read_text())])
        reports = reports.loc[reports.uic_no.isin(wells.uic_number)].copy()
        if reports.id.duplicated().any():
            raise ValueError(f"Duplicate H-10 filing IDs: {name}")
        gas = pd.to_numeric(reports.vol_gas, errors="coerce")
        reports = reports.loc[reports.type_uic.isin(["1", "2"]) & gas.eq(0)]
        reports = reports.rename(columns={"uic_no": "well_id", "formatted_date": "source_date",
                                          "vol_liq": "volume_bbl"})
        yield name, events.loc[events.county.isin(counties)].copy(), reports, "M"


def build(selected=None):
    names = [selected] if selected else list(TEXAS_REGIONS)
    check_sources(SHARED)
    for name, events, reports, frequency in texas():
        if name in names:
            save_region(name, events, reports, frequency, min_training_events=1001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=TEXAS_REGIONS)
    build(parser.parse_args().dataset)
