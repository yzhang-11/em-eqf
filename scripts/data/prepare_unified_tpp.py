"""Build a shared, calendar-indexed event input for temporal point-process models."""
import argparse
from itertools import combinations
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import yaml
from shapely import Polygon, intersects_xy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.prepare_expanded_multiregion import (
    NS_PER_DAY, discrete_b, file_sha256, read_json, screen_overlap, timestamps_ns, write_json,
)

COLUMNS = ["time", "event_id", "magnitude", "latitude", "longitude", "mag_type"]


def read_source(reader, paths):
    if reader == "standard":
        flags = ["quality_screen_pass", "absolute_time_ready"]
        frame = pd.read_parquet(paths[0], columns=COLUMNS + flags)
        for flag in flags:
            if not pd.api.types.is_bool_dtype(frame[flag]) or not frame[flag].fillna(False).all():
                raise ValueError(f"{paths[0]} fails {flag}")
        return frame[COLUMNS]
    if reader == "scedc":
        columns = {"ts": "time", "EVID": "event_id", "M": "mag_type"}
        frame = pd.read_csv(paths[0], usecols=[*columns, "ET", "magnitude", "latitude", "longitude"],
                            dtype={"EVID": "string"}, keep_default_na=False)
        return frame.loc[frame.ET.eq("eq")].rename(columns=columns)[COLUMNS]
    if reader == "comcat":
        columns = {"id": "event_id", "mag": "magnitude", "magType": "mag_type"}
        return pd.concat(
            [pd.read_csv(path, dtype={"id": "string"}).rename(columns=columns)[COLUMNS] for path in paths],
            ignore_index=True,
        )
    if reader == "white":
        with ZipFile(paths[0]) as archive, archive.open("events.csv") as events:
            frame = pd.read_csv(events, sep=r"\s+", dtype={"event_id": "string"})
        # The original file has the time and Ml header labels swapped.
        return frame.rename(columns={"time": "magnitude", "Ml": "time"}).assign(mag_type="ML")[COLUMNS]
    raise ValueError(f"Unknown source reader {reader}")


def select_events(name, frame, spec, cuts):
    source_rows = len(frame)
    frame["event_id"] = frame.event_id.astype("string").str.strip()
    frame["mag_type"] = frame.mag_type.astype("string").str.strip().replace("", pd.NA)
    numeric = ["magnitude", "latitude", "longitude"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    time = pd.to_datetime(frame.time, utc=True, format="mixed", errors="coerce")
    valid = (time.notna() & frame.event_id.notna() & frame.event_id.ne("")
             & np.isfinite(frame[numeric]).all(axis=1)
             & frame.latitude.between(-90, 90) & frame.longitude.between(-180, 180))
    frame = frame.loc[valid].assign(physical_ns=time[valid].array.as_unit("ns").asi8).drop(columns="time")
    valid_rows = len(frame)
    frame = frame.drop_duplicates()
    if frame.event_id.duplicated().any():
        raise ValueError(f"{name}: conflicting rows for the same event ID")
    unique_rows = len(frame)
    m_cut = float(spec["m_cut"])
    if not np.isfinite(m_cut) or not -5 <= m_cut < 10:
        raise ValueError(f"{name}: invalid magnitude cutoff")
    keep = frame.physical_ns.ge(cuts["start"]) & frame.physical_ns.lt(cuts["test_end"])
    if "box" in spec:
        south, north, west, east = spec["box"]
        if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
            raise ValueError(f"{name}: invalid region box")
        keep &= (frame.latitude.ge(south) & frame.latitude.lt(north)
                 & frame.longitude.ge(west) & frame.longitude.lt(east))
    if "polygon" in spec:
        polygon = Polygon(np.load(ROOT / spec["polygon"], allow_pickle=False)[:, ::-1])
        if not polygon.is_valid:
            raise ValueError(f"{name}: invalid region polygon")
        keep &= intersects_xy(polygon, frame.longitude.to_numpy(), frame.latitude.to_numpy())
    events = frame.loc[keep & frame.magnitude.ge(m_cut)].sort_values(["physical_ns", "event_id"])
    if events.empty or events.magnitude.ge(10).any():
        raise ValueError(f"{name}: empty selection or outside the supported magnitude range")
    return events, dict(source_rows=source_rows, invalid_rows=source_rows - valid_rows,
                        identical_duplicates=valid_rows - unique_rows, selected_rows=len(events))


def pack_region(name, events, spec, cuts, calibration, minimum):
    physical_ns = events.physical_ns.to_numpy()
    boundaries = np.array(sorted(set(cuts.values())), dtype=np.int64)
    indices = {key: int(np.searchsorted(physical_ns, date)) for key, date in cuts.items()}
    offset = np.arange(len(events), dtype=np.int64) * 1000
    ns = np.maximum.accumulate(physical_ns - offset) + offset
    shift = int((ns - physical_ns).max())
    if shift > 500_000 or not np.array_equal(np.searchsorted(ns, boundaries),
                                            np.searchsorted(physical_ns, boundaries)):
        raise ValueError(f"{name}: model clock shift exceeds tolerance or crosses a split")
    fit_start = indices.get("train_start", calibration)
    counts = dict(auxiliary=fit_start, train=indices["train_end"] - fit_start,
                  validation=indices["val_end"] - indices["train_end"],
                  test=indices["test_end"] - indices["val_end"])
    if "source_train_end" in indices:
        counts["source_train"] = indices["source_train_end"] - fit_start
        counts["source_validation"] = indices["train_end"] - indices["source_train_end"]
    if fit_start < calibration or any(counts[key] < limit for key, limit in minimum.items()):
        raise ValueError(f"{name}: insufficient coverage for fixed splits: {counts}")
    m_cut = float(spec["m_cut"])
    raw_magnitude = events.magnitude.to_numpy(dtype=np.float64)
    magnitude = m_cut + .1 * np.floor((raw_magnitude - m_cut) / .1 + 1e-8)
    gaps = np.diff(ns[:calibration]) / NS_PER_DAY
    if not 0 < gaps.min() < gaps.max():
        raise ValueError(f"{name}: degenerate calibration history")
    metadata = [m_cut, discrete_b(magnitude[:calibration], m_cut), float(gaps.mean()),
                float(gaps.min()), float(gaps.max()), .1, spec.get("magnitude_type", 0)]
    if "train_start" not in cuts:
        cuts["train_start"] = int(ns[fit_start - 1]) + 1
    region = dict(ns=ns, physical_ns=physical_ns, magnitude=magnitude, raw_magnitude=raw_magnitude,
                  event_id=events.event_id.tolist(), latitude=events.latitude.to_numpy(dtype=np.float64),
                  longitude=events.longitude.to_numpy(dtype=np.float64),
                  mag_type=events.mag_type.fillna("").tolist(), metadata=metadata, **indices)
    region.update(train_start=fit_start, calibration_end=calibration, time_bounds_ns=cuts)
    audit = dict(counts=counts, metadata=metadata, max_clock_shift_ns=shift,
                 tied_events=int(np.count_nonzero(np.diff(physical_ns) == 0)),
                 calibration_last_ns=int(physical_ns[calibration - 1]), time_bounds_ns=cuts)
    return region, audit


def specifications(config, group):
    if group == "benchmark":
        for name, spec in config[group].items():
            paths = [ROOT / spec["source"]] if "source" in spec else sorted(Path(spec["source_dir"]).glob("*.csv"))
            if "years" in spec and [path.stem for path in paths] != list(map(str, range(*spec["years"]))):
                raise ValueError(f"{name}: annual source inventory is incomplete")
            yield name, spec, paths, spec["dates"], config["benchmark_min_events"]
        return
    previous = read_json(ROOT / config["multiregion_reference"])
    settings = previous["settings"]
    for name, audit in sorted(previous["regions"].items()):
        path = Path(audit["source"])
        if file_sha256(path) != audit["source_sha256"]:
            raise ValueError(f"{name}: regional source differs from the audited input")
        spec = dict(audit["selection"], reader="standard", m_cut=audit["metadata"][0])
        dates = {key: settings[key] for key in ("source_train_end", "train_end", "val_end", "test_end")}
        dates["start"] = spec.get("start", audit["first_event_utc"])
        yield name, spec, [path], dates, settings["min_events"]


def prepare(config, group, output):
    calibration = config["calibration_events"]
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    regions, audits, frames = {}, {}, {}
    for name, spec, paths, dates, minimum in specifications(config, group):
        keys = ["start", "train_start", "source_train_end", "train_end", "val_end", "test_end"]
        ordered = [key for key in keys if key in dates]
        cuts = dict(zip(ordered, map(int, timestamps_ns([dates[key] for key in ordered]))))
        if not np.all(np.diff(list(cuts.values())) > 0):
            raise ValueError(f"{name}: invalid time bounds")
        events, counts = select_events(name, read_source(spec["reader"], paths), spec, cuts)
        frames[name] = events
        regions[name], audit = pack_region(name, events, spec, cuts, calibration, minimum)
        inputs = paths + ([ROOT / spec["polygon"]] if "polygon" in spec else [])
        audits[name] = dict(audit, **counts, source_files={str(path): file_sha256(path) for path in inputs},
                            dates=dates, selection=spec)
    pairs = list(combinations(frames, 2)) if group == "multiregion" else []
    for left, right in pairs:
        if left.startswith("USGS_") and right.startswith("USGS_"):
            shared = set(frames[left].event_id) & set(frames[right].event_id)
            if shared:
                raise ValueError(f"Shared USGS event ID: {left}/{right}")
        if hit := screen_overlap(frames[left], frames[right]):
            raise ValueError(f"Possible cross-region duplicate {left}/{right}: {hit}")
    vocabulary = {"<unknown>": 0, **{name: i for i, name in enumerate(regions, 1)}}
    event_fields = ("ns", "physical_ns", "magnitude", "raw_magnitude", "event_id",
                    "latitude", "longitude", "mag_type")
    development = {}
    for name, region in regions.items():
        end = region["val_end"]
        development[name] = {key: value[:end] if key in event_fields else value
                             for key, value in region.items() if key != "test_end"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".catalog-build-") as temporary:
        staging = Path(temporary)
        for filename, subset in (("development.pt", development), ("test.pt", regions)):
            torch.save(dict(regions=subset, vocabulary=vocabulary, calibration_events=calibration),
                       staging / filename)
        write_json(staging / "dataset.json", dict(
            schema="unified-tpp-v1", group=group, calibration_events=calibration, regions=audits,
            overlap_pairs_checked=len(pairs), preparer_sha256=file_sha256(__file__),
            helper_sha256=file_sha256(ROOT / "scripts/data/prepare_expanded_multiregion.py"),
            artifacts={path.name: file_sha256(path) for path in sorted(staging.iterdir())},
        ))
        staging.rename(output)
    for name, audit in audits.items():
        print(name, audit["counts"], "clock_shift_ns", audit["max_clock_shift_ns"])
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group", choices=("benchmark", "multiregion"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.group, args.output)
