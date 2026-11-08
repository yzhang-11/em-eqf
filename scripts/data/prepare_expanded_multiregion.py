"""Index audited catalog snapshots and prepare a versioned TPP cohort."""
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.catalogs.external_catalog import file_sha256

REVISION = ROOT / "experiments/revision"
STANDARD_REPORTS = (
    "catalog_standardization_20261105",
    "catalog_additions_20261105",
    "catalog_more_20261105",
    "catalog_research_20261106",
)
NS_PER_DAY = 86_400_000_000_000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(study, filename):
    report = REVISION / study / "reports" / filename
    with report.open(encoding="utf-8", newline="") as handle:
        return [dict(row, evidence=str(report.relative_to(ROOT))) for row in csv.DictReader(handle)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def catalog_index(settings):
    rows = []
    for study in STANDARD_REPORTS:
        for row in read_rows(study, "catalogs.csv"):
            name = row["dataset"]
            if row.get("dataloader_verified", "").lower() != "true":
                raise ValueError(f"Unverified standard catalog: {name}")
            selected = name in settings["regions"]
            reason = settings["separate_catalogs"].get(name)
            if selected == bool(reason):
                raise ValueError(f"Give {name} one explicit cohort decision")
            rows.append(dict(row, group="cohort" if selected else "standalone", reason=reason))
    for row in read_rows("catalog_more_20261105", "staged_catalogs.csv"):
        rows.append(dict(row, group="staged"))
    for row in read_rows("catalog_benchmarks_20261106", "benchmark_catalogs.csv"):
        rows.append(dict(row, group="benchmark", reason="Author split or compatibility variant"))
    historical = REVISION / "multiregion_pretraining/reports/chuandian_scedc_tpp_20261105/data.json"
    for name, row in read_json(historical)["targets"].items():
        rows.append(dict(dataset=name, group="historical", reason="Frozen target protocol", **row))
    names = [row["dataset"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate dataset in catalog index")
    for row in rows:
        source = Path(row.get("source") or row.get("path") or "")
        row["source_available"] = source.is_file()
        if row.get("source_sha256"):
            if not source.is_file() or file_sha256(source) != row["source_sha256"]:
                raise ValueError(f"Source missing or changed: {row['dataset']} ({source})")
    return {row["dataset"]: row for row in rows}


def timestamps_ns(values):
    times = pd.to_datetime(values, utc=True, format="mixed", errors="raise")
    if times.hasnans:
        raise ValueError("Missing physical timestamp")
    return times.as_unit("ns").asi8


def load_events(name, spec, entry, end):
    columns = ["time", "event_id", "magnitude", "latitude", "longitude",
               "quality_screen_pass", "absolute_time_ready"]
    frame = pd.read_parquet(entry["source"], columns=columns)
    for key in ("quality_screen_pass", "absolute_time_ready"):
        if not pd.api.types.is_bool_dtype(frame[key]) or not frame[key].fillna(False).all():
            raise ValueError(f"{name}: source fails {key}")
    if frame[columns].isna().any().any():
        raise ValueError(f"{name}: null event fields")
    frame["event_id"] = frame.event_id.astype(str).str.strip()
    if frame.event_id.eq("").any() or frame.event_id.duplicated().any():
        raise ValueError(f"{name}: empty or duplicate event IDs")
    if not np.isfinite(frame[["magnitude", "latitude", "longitude"]]).all().all():
        raise ValueError(f"{name}: nonfinite event fields")
    if not frame.latitude.between(-90, 90).all() or not frame.longitude.between(-180, 180).all():
        raise ValueError(f"{name}: invalid coordinates")
    frame["physical_ns"] = timestamps_ns(pd.Index(frame.time))
    mc = float(entry["mc"])
    if not np.isfinite(mc) or not -5 <= mc < 10:
        raise ValueError(f"{name}: invalid Mc {mc}")
    keep = (frame.physical_ns < end) & (frame.magnitude >= mc)
    if "start" in spec:
        keep &= frame.physical_ns >= timestamps_ns([spec["start"]])[0]
    if "box" in spec:
        lat_min, lat_max, lon_min, lon_max = spec["box"]
        if not (-90 <= lat_min < lat_max <= 90 and -180 <= lon_min < lon_max <= 180):
            raise ValueError(f"{name}: invalid region box")
        keep &= (frame.latitude.ge(lat_min) & frame.latitude.lt(lat_max)
                 & frame.longitude.ge(lon_min) & frame.longitude.lt(lon_max))
    events = frame.loc[keep].sort_values(["physical_ns", "event_id"]).reset_index(drop=True)
    if events.empty or events.magnitude.ge(10).any():
        raise ValueError(f"{name}: empty selection or M >= model support bound")
    return events, mc, len(frame)


def screen_overlap(left, right):
    """Prior audit rule: within 2 seconds, 15 km and one magnitude unit."""
    left_time = left.physical_ns.to_numpy()
    right_time = right.physical_ns.to_numpy()
    starts = np.searchsorted(right_time, left_time - 2_000_000_000)
    stops = np.searchsorted(right_time, left_time + 2_000_000_000, side="right")
    left_lat, left_lon = np.radians(left[["latitude", "longitude"]].to_numpy().T)
    right_lat, right_lon = np.radians(right[["latitude", "longitude"]].to_numpy().T)
    left_mag, right_mag = left.magnitude.to_numpy(), right.magnitude.to_numpy()
    for i in np.flatnonzero(stops > starts):
        span = slice(starts[i], stops[i])
        haversine = (np.sin((right_lat[span] - left_lat[i]) / 2) ** 2
                     + np.cos(left_lat[i]) * np.cos(right_lat[span])
                     * np.sin((right_lon[span] - left_lon[i]) / 2) ** 2)
        distance = 12_742 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        hits = np.flatnonzero((distance <= 15) & (np.abs(right_mag[span] - left_mag[i]) <= 1))
        if hits.size:
            return left.event_id.iloc[i], right.event_id.iloc[starts[i] + hits[0]]
    return None


def discrete_b(magnitude, mc):
    mean_bin = float(np.mean((magnitude - mc) / 0.1))
    return float(np.log1p(1 / max(mean_bin, 1e-9)) / (0.1 * np.log(10)))


def region_payload(name, events, mc, spec, settings, cuts):
    physical_ns = events.physical_ns.to_numpy()
    source_train, train, val = np.searchsorted(physical_ns, cuts[:3]).tolist()
    end = len(events)
    calibration = settings["calibration_events"]
    counts = {
        "source_train": source_train - calibration,
        "source_validation": train - source_train,
        "train": train - calibration,
        "validation": val - train,
        "test": end - val,
    }
    minimum = settings["min_events"]
    if any(count < minimum[key] for key, count in counts.items()):
        raise ValueError(f"{name}: insufficient events for fixed splits: {counts}")
    # Preserve exact nanoseconds and order. Unfold ties only on the model clock.
    offset = np.arange(end, dtype=np.int64) * 1000
    model_ns = np.maximum.accumulate(physical_ns - offset) + offset
    shift = int((model_ns - physical_ns).max())
    if shift > 500_000 or not np.array_equal(np.searchsorted(model_ns, cuts), np.searchsorted(physical_ns, cuts)):
        raise ValueError(f"{name}: numerical clock shifts a split or exceeds 0.5 ms")
    raw_magnitude = events.magnitude.to_numpy(dtype=np.float64)
    bins = np.floor((raw_magnitude - mc) / .1 + 1e-8)
    magnitude = mc + .1 * bins
    gaps = np.diff(model_ns[:calibration]) / NS_PER_DAY
    if gaps.min() <= 0 or gaps.min() >= gaps.max():
        raise ValueError(f"{name}: degenerate calibration gaps")
    metadata = [mc, discrete_b(magnitude[:calibration], mc), float(gaps.mean()),
                float(gaps.min()), float(gaps.max()), .1, spec.get("magnitude_type", 0)]
    region = dict(ns=model_ns, physical_ns=physical_ns, magnitude=magnitude,
                  raw_magnitude=raw_magnitude, event_id=events.event_id.tolist(), metadata=metadata,
                  source_train_end=source_train, train_end=train, val_end=val, test_end=end)
    audit = dict(counts=counts, metadata=metadata, max_clock_shift_ns=shift,
                 tied_events=int(np.count_nonzero(np.diff(physical_ns) == 0)),
                 calibration_last_utc=str(pd.Timestamp(physical_ns[calibration - 1], tz="UTC")),
                 first_event_utc=str(pd.Timestamp(physical_ns[0], tz="UTC")),
                 last_event_utc=str(pd.Timestamp(physical_ns[-1], tz="UTC")))
    return region, audit


def prepare(settings, output):
    if output.exists():
        manifest = read_json(output / "dataset.json")
        if manifest["settings"] != settings or manifest["preparer_sha256"] != file_sha256(__file__):
            raise ValueError("Use a new output directory for changed data settings or code")
        for name, digest in manifest["artifacts"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"Prepared artifact changed: {name}")
        return manifest

    cuts = timestamps_ns([settings[key] for key in ("source_train_end", "train_end", "val_end", "test_end")])
    if not np.all(np.diff(cuts) > 0) or settings["calibration_events"] < 3:
        raise ValueError("Expected increasing cutoff dates and at least three calibration events")
    if set(settings["min_events"]) != {"source_train", "source_validation", "train", "validation", "test"}:
        raise ValueError("Provide all five event-count limits")
    if min(settings["min_events"].values()) < 1 or len(settings["regions"]) < 2:
        raise ValueError("Expected positive event-count limits and at least two regions")

    index = catalog_index(settings)
    regions, frames, audits = {}, {}, {}
    for name, spec in sorted(settings["regions"].items()):
        entry = index[name]
        frames[name], mc, source_rows = load_events(name, spec, entry, cuts[-1])
        regions[name], audit = region_payload(name, frames[name], mc, spec, settings, cuts)
        audits[name] = dict(audit, source=entry["source"], source_sha256=entry["source_sha256"],
                            source_rows=source_rows, selected_rows=len(frames[name]), selection=spec)

    pairs = list(combinations(regions, 2))
    for left, right in pairs:
        if left.startswith("USGS_") and right.startswith("USGS_"):
            shared = set(frames[left].event_id) & set(frames[right].event_id)
            if shared:
                raise ValueError(f"Shared USGS ID {left}/{right}: {min(shared)}")
        hit = screen_overlap(frames[left], frames[right])
        if hit is not None:
            raise ValueError(f"Possible cross-region duplicate {left}/{right}: {hit}")

    vocabulary = {"<unknown>": 0, **{name: i for i, name in enumerate(regions, 1)}}
    fit_regions = {}
    for name, region in regions.items():
        fit_regions[name] = dict(region)
        for key in ("ns", "physical_ns", "magnitude", "raw_magnitude", "event_id"):
            fit_regions[name][key] = region[key][:region["val_end"]].copy()
        fit_regions[name].pop("test_end")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".catalog-build-") as temporary:
        staging = Path(temporary)
        for filename, subset in (("development.pt", fit_regions), ("test.pt", regions)):
            torch.save(dict(regions=subset, vocabulary=vocabulary,
                            calibration_events=settings["calibration_events"]), staging / filename)
        write_json(staging / "catalog_index.json", list(index.values()))
        manifest = dict(schema="expanded-regional-tpp-v1", settings=settings, regions=audits,
                        overlap_pairs_checked=len(pairs), overlap_candidates=0,
                        preparer_sha256=file_sha256(__file__),
                        artifacts={p.name: file_sha256(p) for p in sorted(staging.iterdir())})
        write_json(staging / "dataset.json", manifest)
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["paths"]["data"]
    manifest = prepare(config["data"], output)
    for name, row in manifest["regions"].items():
        print(name, row["counts"])
    print(f"{len(manifest['regions'])} regions, {manifest['overlap_pairs_checked']} pairs checked: {output}")


if __name__ == "__main__":
    main()
