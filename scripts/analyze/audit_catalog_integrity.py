"""Read-only source, coverage and completeness audit for em_eqf catalog snapshots."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

CORE = ["event_id", "time", "magnitude", "latitude", "longitude", "depth_km"]
OPTIONAL = {
    "mag_type": ("mag_type", "magnitude_type"),
    "event_type": ("event_type", "class"),
    "review_status": ("evaluation_status", "source_review_status", "source_status"),
    "evaluation_mode": ("evaluation_mode",),
    "station_count": ("station_count",),
    "phase_count": ("phase_count", "N_obs", "Nphs", "number_picks"),
    "azimuthal_gap": ("azimuthal_gap", "gap_deg", "GAP", "Gap"),
    "rms": ("rms", "rms_s", "RMS", "source_rms"),
    "horizontal_error_km": ("horizontal_error_km", "erh_km", "ellipse_larger_km", "max_horizontal_uncertainty", "errH"),
    "depth_error_km": ("depth_error_km", "erz_km", "z_error", "vertical_uncertainty", "errZ"),
    "magnitude_error": ("magnitudeuncertainty", "magError", "ML_unc", "StdMl"),
    "magnitude_station_count": ("magnitudestationcount", "magNst", "ML_stations"),
}
NUMERIC = set(CORE[2:] + list(OPTIONAL)[4:])
JMA_BOXES = dict(JMA_Kyushu=(30,34,129,133), JMA_WestHonshu=(34,37,130,137),
                 JMA_EastHonshu=(34,41,137,145), JMA_Hokkaido=(41,46,139,146))
BATCHES = ["catalog_standardization_20261105", "catalog_additions_20261105",
           "catalog_more_20261105", "catalog_research_20261106",
           "catalog_source_expansion_20261106", "catalog_benchmarks_20261106"]
# Additional columns needed to reconstruct recent quality-only masks.
MASK_COLUMNS = {"source_year", "source_review_status", "source_quality", "depth_control",
    "station_count", "gap_deg", "rms_s", "erh_km", "erz_km", "phase_count", "N_obs",
    "GAP", "RMS", "ellipse_larger_km", "z_error", "rms", "airgun", "mislocated",
    "growclust_relocated", "number_picks", "number_p_and_s_picks", "azimuthal_gap",
    "max_horizontal_uncertainty", "vertical_uncertainty", "Nphs", "Gap", "errH", "errZ", "class"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc(values):
    return pd.to_datetime(values, utc=True, format="mixed", errors="coerce").dt.as_unit("ns")


def canonical(frame):
    frame = frame.rename(columns={"id": "event_id"} if "event_id" not in frame and "id" in frame else {})
    if "time" in frame:
        parsed = utc(frame.time)
        frame["_invalid_time"] = parsed.isna() & frame.time.notna() & frame.time.astype("string").str.strip().ne("")
        frame["time"] = parsed
    if "event_id" in frame:
        frame["event_id"] = frame.event_id.astype("string")
    return frame


def read_source(path):
    if path.suffix == ".csv":
        from src.catalogs.earthquakenpp_20261106 import read_author_csv
        return canonical(read_author_csv(path))
    cols = set(CORE) | MASK_COLUMNS | {"quality_screen_pass"}
    cols.update(c for names in OPTIONAL.values() for c in names)
    present = pq.ParquetFile(path).schema_arrow.names
    return canonical(pd.read_parquet(path, columns=[c for c in present if c in cols]))


def inventory(root):
    base = root / "experiments/revision"
    for batch in BATCHES:
        for row in load(base / batch / "reports/summary.json")["datasets"]:
            yield row["dataset"], batch, Path(row["cache"]), Path(row["source"])
    batch = "catalog_doi_expansion_20261106"
    for report in sorted((base / batch / "reports").glob("*.json")):
        row = load(report)
        if row.get("dataloader_verified") and "raw_path" in row:
            cache = Path(row["cache"])
            yield report.stem, batch, cache, Path(load(cache / "processing_report.json")["source"])


def raw_path(source):
    for filename in ["normalized_all.parquet", "normalised_20261106.parquet",
                     "normalised_20261105.parquet", "earthquakes_20261105.parquet"]:
        candidate = source.with_name(filename)
        if candidate.is_file():
            return candidate
    return source


def recent_quality(name, f):
    if name in {"Taiwan_CWA", "Taiwan_Wu2008", "Turkiye_EAFZ"}:
        spatial = f.gap_deg.between(0, 180) & f.erh_km.between(0, 2) & f.erz_km.between(0, 5)
        if name == "Turkiye_EAFZ":
            return spatial & f.phase_count.ge(12)
        spatial &= f.station_count.ge(8) & f.rms_s.between(0, 1)
        spatial &= f.latitude.between(20, 27) & f.longitude.between(118, 124) & f.depth_km.between(0, 700)
        if name == "Taiwan_CWA":
            return spatial & f.source_review_status.eq("F")
        return spatial & f.depth_control.eq("F") & f.source_quality.isin(["A", "B"])
    if name == "Chile_Potin":
        return f.mag_type.eq("l") & f.rms.between(0, .5) & f.depth_km.between(-5, 700)
    if name == "Chile_Atacama":
        return (f["class"].isin(["interface", "intraslab", "upper_plate", "outer_rise"])
                & f.airgun.eq(False) & f.mislocated.eq(False) & f.growclust_relocated.eq(True)
                & f.number_picks.ge(12) & f.number_p_and_s_picks.ge(5) & f.azimuthal_gap.between(0, 180)
                & f.max_horizontal_uncertainty.between(0, 2) & f.vertical_uncertainty.between(0, 5))
    if name == "Greece_WestHellenic":
        return (f.N_obs.ge(16) & f.GAP.between(0, 240) & f.RMS.between(0, .35)
                & f.ellipse_larger_km.between(0, 5) & f.z_error.between(0, 5))
    if name == "Greece_SantoriniSwarm":
        return (f.Nphs.ge(12) & f.Gap.between(0, 180) & f.RMS.between(0, .35)
                & f.errH.between(0, 2) & f.errZ.between(0, 5))
    return None


def missing_rows(frame, identity, stage, depth_optional=False):
    for key in CORE + list(OPTIONAL):
        choices = OPTIONAL.get(key, (key,))
        col = next((col for col in choices if col in frame), None)
        required = key in CORE and not (key == "depth_km" and depth_optional)
        if col is None:
            yield dict(dataset=identity, stage=stage, field=key, source_column=None, required=required,
                       records=len(frame), missing=len(frame), invalid=0, status="column_absent")
            continue
        values = frame[col]
        null = values.isna()
        if pd.api.types.is_string_dtype(values):
            null |= values.astype("string").str.strip().isin(["", "nan", "null", "None", "N/A"])
        invalid = pd.Series(False, index=values.index)
        if key == "time" and "_invalid_time" in frame:
            invalid = frame["_invalid_time"]
            null &= ~invalid
        if key in NUMERIC:
            num = pd.to_numeric(values, errors="coerce")
            invalid = ~np.isfinite(num)
            if key in {"latitude", "longitude"}:
                bound = 90 if key == "latitude" else 180
                invalid |= ~num.between(-bound, bound)
            if key in {"station_count", "phase_count", "azimuthal_gap", "rms", "horizontal_error_km", "depth_error_km"}:
                invalid |= num.lt(0)
            if key == "azimuthal_gap":
                invalid |= num.gt(360)
        yield dict(dataset=identity, stage=stage, field=key, source_column=col, required=required,
                   records=len(frame), missing=int(null.sum()), invalid=int((invalid & ~null).sum()),
                   status="present")


def fmd(values, width=.1, origin=0.):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([]), np.array([], dtype=int)
    bins = np.floor((values-origin)/width + 1e-9).astype(int)
    low = bins.min()
    count = np.bincount(bins - low)
    return np.round(origin + (np.arange(len(count)) + low)*width, 8), count


def gr_fit(values, cutoff, width=.1):
    values = np.asarray(values, dtype=float)
    tail = values[np.isfinite(values) & (values >= cutoff - 1e-9)]
    if len(tail) < 100:
        return len(tail), None, None
    bins = np.maximum(np.floor((tail-cutoff)/width+1e-9).astype(int), 0)
    hist = np.bincount(bins)
    if len(hist) < 6 or np.count_nonzero(hist) < 5:
        return len(tail), None, None
    mean_bin = bins.mean()
    q = mean_bin/(mean_bin + 1)
    expected = len(tail)*q**np.arange(len(hist))
    observed = np.cumsum(hist[::-1])[::-1]
    score = 100*(1 - np.abs(expected-observed).sum()/observed.sum())
    return len(tail), float(score), float(-np.log10(q)/width)


def completeness(values, cutoff):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    row = {"n": len(values), "m_min": None if not len(values) else float(values.min()),
           "below_model_cutoff": int((values < cutoff - 1e-9).sum())}
    n, fit, b = gr_fit(values, cutoff)
    row.update(n_at_cutoff=n, gft_at_cutoff=fit, b_at_cutoff=b)
    if len(values) < 100:
        return row
    for width in [.05, .1, .2]:
        edges, count = fmd(values, width)
        row[f"maxc_edge_{width:g}"] = float(edges[count.argmax()])
    edges, counts = fmd(values)
    # Scan at and above the peak. First crossings are diagnostics, not a proof of completeness.
    for edge in edges[counts.argmax():]:
        _, score, _ = gr_fit(values, float(edge))
        if score is None:
            continue
        for criterion in (90, 95):
            if score >= criterion:
                row.setdefault(f"gft{criterion}_edge", float(edge))
        if "gft95_edge" in row:
            break
    return row


def time_counts(frame, boundaries):
    times = np.sort(frame.time.dropna().array.as_unit("ns").asi8)
    indices = np.searchsorted(times, pd.DatetimeIndex(boundaries).as_unit("ns").asi8)
    return np.diff(indices)


def cover(name, raw, quality, selected, start, end, cut):
    days = (end-start).total_seconds()/86400
    resolution = "D" if days <= 90 else "MS"
    grid = pd.date_range(start.floor("D") if resolution == "D" else start.normalize().replace(day=1),
                         end, freq=resolution)
    boundaries = pd.DatetimeIndex([start, *grid[(grid > start) & (grid < end)], end])
    window = pd.DataFrame(dict(dataset=name, frequency=resolution, start=boundaries[:-1], end=boundaries[1:]))
    for label, data in [("source",raw),("source_at_cut",raw.loc[raw.magnitude.ge(cut)]),
                        ("quality",quality),("standard",selected)]:
        window[label] = time_counts(data, boundaries)
    if "mag_type" in raw and "mag_type" in selected:
        types = set(selected.mag_type.dropna())
        same = raw.loc[raw.mag_type.isin(types) & raw.magnitude.ge(cut)] if types else raw.loc[raw.magnitude.ge(cut)]
    else:
        same = raw.loc[raw.magnitude.ge(cut)]
    window["source_same_type_at_cut"] = time_counts(same, boundaries)
    window["exposure_days"] = np.diff(boundaries.as_unit("ns").asi8)/86400e9
    window["source_per_day"] = window.source/window.exposure_days
    window["standard_per_day"] = window.standard/window.exposure_days
    window["status"] = np.select(
        [window.standard.gt(0), window.source.eq(0), window.source_at_cut.eq(0), window.source_same_type_at_cut.eq(0)],
        ["has_standard", "no_source_events", "no_source_at_cutoff", "no_same_type_at_cutoff"],
        default="no_standard_after_screen")
    return window


def gaps(name, raw, selected, start, end, cut):
    # A physical gap is bounded by consecutive distinct times or the window edges.
    points = pd.DatetimeIndex([start, *selected.time.dropna().unique(), end]).sort_values().unique()
    durations = np.diff(points.as_unit("ns").asi8)
    indices = np.argsort(-durations, kind="stable")[:15]
    series = [np.sort(f.time.dropna().array.as_unit("ns").asi8) for f in (raw, raw.loc[raw.magnitude.ge(cut)])]
    rows = []
    for i in indices:
        left, right = points[i], points[i+1]
        counts = [int(np.searchsorted(t, right.value)-np.searchsorted(t,left.value,side="right")) for t in series]
        rows.append(dict(dataset=name, start=left, end=right, days=durations[i]/86400e9,
                         edge=bool(i == 0 or i == len(points)-2), source_inside=counts[0],
                         source_at_cut_inside=counts[1], status=("source_events_present" if counts[0] else "source_empty")))
    return rows


def cache_integrity(root, name, cache, cfg, source, frame):
    from src.catalogs.external_catalog import file_sha256
    from src.data import Catalog, TppDataset
    assert file_sha256(source) == cfg["source_sha256"], name
    seq = TppDataset.load_from_disk(cache/"full_sequence.pt")[0]
    ref = read_source(source).set_index("event_id")
    common = ref.loc[frame.event_id]
    differences = {}
    for key in CORE[1:]:
        if key in common and key in frame:
            differences[key] = int(np.count_nonzero(common[key].to_numpy() != frame[key].to_numpy()))
    assert not any(differences.values()), (name, differences)
    origin = pd.Timestamp(cfg["start_ts"], tz="UTC").value
    physical_days = (frame.time.array.asi8-origin)/86400e9
    changes = frame.time_adjustment_seconds.to_numpy()
    assert np.max(np.abs(seq.arrival_times.numpy()-physical_days-changes/86400))*86400 < 1e-6
    assert np.max(np.abs(changes)) <= float(cfg["timestamp_resolution_seconds"])/2 + 1e-7
    assert seq.inter_times.dtype == torch.float64 and bool((seq.inter_times[:-1] > 0).all())
    assert bool(torch.isfinite(seq.inter_times).all())
    assert np.array_equal(seq.mag.numpy(), frame.magnitude.to_numpy(dtype=np.float32))
    obj = Catalog.by_name(name+"-Standard")()
    assert obj.root_dir.resolve() == cache.resolve(), name
    split_rows = {}
    starts = [cfg[k] for k in ["train_start_ts","val_start_ts","test_start_ts","end_ts"]]
    for split, lo, hi in zip(["train","val","test"], starts, starts[1:]):
        part = getattr(obj, split)[0]
        mask = part.arrival_times >= part.t_nll_start
        actual = part.event_index[mask].numpy()
        expected = np.flatnonzero(frame.time.ge(lo) & frame.time.lt(hi))
        assert np.array_equal(actual, expected), (name, split)
        split_rows[split] = len(expected)
    train = frame.loc[frame.time.ge(starts[0]) & frame.time.lt(starts[1])]
    stats = torch.load(cache/"norm_stats.pt", weights_only=False)
    columns = [("x","x"),("y","y")] if name.startswith("ENPP_") else [
        ("latitude","latitude"),("longitude","longitude"),("depth","depth_km")]
    for key, col in columns:
        lo, hi = float(train[col].min()), float(train[col].max())
        assert stats[key] == dict(min=lo, max=hi), (name, key, stats[key])
    return dict(dataset=name, cache=str(cache), physical_differences=differences,
                exact_duplicate_physical_rows=int(frame.duplicated([k for k in CORE[1:] if k in frame]).sum()),
                tied_physical_rows=int(frame.time.duplicated().sum()), shifted_model_rows=int(np.count_nonzero(changes)),
                max_model_shift_seconds=float(np.max(np.abs(changes))),
                max_float32_magnitude_error=float(np.max(np.abs(seq.mag.numpy().astype(float)-frame.magnitude))),
                nonpositive_model_event_intervals=0, train_only_norm=True, split_counts=split_rows)


def diagnostics(name, raw, quality, selected, cfg, rows, fmd_rows, strata_rows):
    cut = float(cfg["mag_completeness"])
    boundaries = [pd.Timestamp(cfg[k], tz="UTC") for k in ["train_start_ts","val_start_ts","test_start_ts","end_ts"]]
    periods = [("all_scored", boundaries[0], boundaries[-1])]
    periods += list(zip(["train","val","test"], boundaries, boundaries[1:]))
    # Calendar subperiods expose network and event-rate changes hidden by pooled fits.
    days = (boundaries[-1]-boundaries[0]).days
    freq = "7D" if days < 70 else "MS" if days < 365 else "YS"
    step = pd.date_range(boundaries[0], boundaries[-1], freq=freq)
    dates = sorted(set([boundaries[0],*step,boundaries[-1]]))
    periods += [(lo.strftime("%Y-%m-%d"),lo,hi) for lo,hi in zip(dates,dates[1:])]
    for label, stage in [("source",raw),("quality",quality),("standard",selected)]:
        for period, lo, hi in periods:
            f = stage.loc[stage.time.ge(lo) & stage.time.lt(hi)]
            rows.append(dict(dataset=name, stage=label, period=period, start=lo, end=hi, cutoff=cut,
                count=len(f), **completeness(f.magnitude.to_numpy(), cut)))
        edges, counts = fmd(stage.magnitude.to_numpy())
        fmd_rows.extend(dict(dataset=name, stage=label, magnitude_edge=x, count=n) for x,n in zip(edges,counts))
    # Use physical depth strata within the quality cohort, kept separate from the pooled result.
    for label, lo, hi in [("lt30", -np.inf, 30), ("30to70",30,70), ("ge70",70,np.inf)]:
        if "depth_km" in quality:
            f = quality.loc[quality.depth_km.ge(lo)&quality.depth_km.lt(hi)]
            strata_rows.append(dict(dataset=name, depth_band=label, **completeness(f.magnitude.to_numpy(),cut)))


def read_legacy(root):
    from src.catalogs.external_catalog import file_sha256
    report = load(root/"experiments/revision/multiregion_pretraining/reports/chuandian_scedc_tpp_20261105/data.json")
    target = Path(report["target_path"])
    assert file_sha256(target) == report["target_sha256"]
    packed = torch.load(target, map_location="cpu", weights_only=False)
    for key, spec in report["targets"].items():
        path = Path(spec["source"])
        assert file_sha256(path) == spec["source_sha256"]
        raw = pd.read_csv(path).rename(columns={"ts":"time","Magnitude":"magnitude","Depth":"depth_km",
            "depth":"depth_km","Latitude":"latitude","Longitude":"longitude","EVID":"event_id","M":"mag_type","ET":"event_type"})
        # The SCEDC export contains both original magnitude and a reference Magnitude column.
        raw = raw.loc[:,~raw.columns.duplicated(keep="last")]
        if "event_id" not in raw:
            raw["event_id"] = raw.index.astype(str)
        raw = canonical(raw)
        region = packed["regions"][key]
        ids = [value.split(":",1)[1] for value in region["event_id"]]
        selected = raw.set_index("event_id").loc[ids].reset_index()
        assert np.array_equal(selected.time.array.asi8, region["physical_ns"])
        assert np.array_equal(selected.magnitude.to_numpy(), region["raw_magnitude"])
        for field in ("latitude","longitude"):
            assert np.array_equal(selected[field].to_numpy(),region[field])
        shifted = region["ns"]-region["physical_ns"]
        delta = region["magnitude"]-region["raw_magnitude"]
        assert np.all(np.diff(region["ns"])>0) and np.all(shifted>=0)
        cfg = dict(zip(["start_ts","val_start_ts","test_start_ts","end_ts"], spec["dates"]))
        cfg.update(mag_completeness=spec["mc"],train_start_ts=str(selected.time.iloc[spec["calibration_events"]].tz_localize(None)))
        yield "Target_"+key, raw, selected, cfg, dict(dataset="Target_"+key, cache=str(target),
            physical_differences=dict(time=0,magnitude=0,latitude=0,longitude=0),
            shifted_model_rows=int(np.count_nonzero(shifted)), max_model_shift_seconds=float(shifted.max()/1e9),
            binned_model_magnitude_rows=int(np.count_nonzero(np.abs(delta)>1e-8)),
            max_magnitude_binning_change=float(np.abs(delta).max()))


def run(root, out, requested):
    sys.path.insert(0, str(root))
    out.mkdir(parents=True, exist_ok=True)
    missing, rate, largest_gaps, mc, curves, strata, integrity, overview = [],[],[],[],[],[],[],[]
    all_items = list(inventory(root))
    items = []
    for name, batch, cache, source in all_items:
        if requested and name not in requested:
            continue
        cfg = load(cache/"catalog_cfg.json")
        frame = pd.read_parquet(cache/"catalog.parquet")
        if "event_id" not in frame:
            frame = frame.rename(columns={"id":"event_id"})
        frame["time"] = utc(frame.ts)
        audit = cache_integrity(root, name, cache, cfg, source, frame)
        location = raw_path(source)
        raw = read_source(location)
        if name in JMA_BOXES:
            a,b,c,d=JMA_BOXES[name]
            raw=raw.loc[raw.latitude.ge(a)&raw.latitude.lt(b)&raw.longitude.ge(c)&raw.longitude.lt(d)]
        independent = recent_quality(name,raw)
        if independent is None:
            quality = read_source(source) if location != source else raw
            basis = "upstream quality snapshot; source floors may already apply" if location != source else "author snapshot; prepublication thresholds unknown"
        else:
            physical = ["magnitude","latitude","longitude","depth_km"]
            quality = raw.loc[independent & np.isfinite(raw[physical]).all(axis=1)]
            basis = "quality reconstructed before applying model cutoff"
        items.append((name,batch,raw,quality,frame,cfg,audit,str(location),basis))
        print("INTEGRITY", name, len(frame), flush=True)
    if not requested or any(n.startswith("Target_") for n in requested):
        for name,raw,selected,cfg,audit in read_legacy(root):
            if not requested or name in requested:
                items.append((name,"frozen_main_targets",raw,raw,selected,cfg,audit,None,"frozen source input; source coverage may vary"))
    for name,batch,raw,quality,selected,cfg,audit,location,basis in items:
        lo,hi = (pd.Timestamp(cfg[k], tz="UTC") for k in ["start_ts","end_ts"])
        cut = float(cfg["mag_completeness"])
        scoped_raw = raw.loc[raw.time.ge(lo)&raw.time.lt(hi)]
        scoped_quality = quality.loc[quality.time.ge(lo)&quality.time.lt(hi)]
        if basis.startswith("quality reconstructed"):
            expected = set(scoped_quality.loc[scoped_quality.magnitude.ge(cut),"event_id"])
            absent = expected-set(selected.event_id)
            added = set(selected.event_id)-expected
            if name == "Turkiye_EAFZ":
                alias = root/"experiments/revision/catalog_source_expansion_20261106/reports/eafz_identical_physical_records.csv"
                assert absent == set(pd.read_csv(alias).alias_id)
            else:
                assert not absent, (name,len(absent))
            assert not added
        depth_optional = name.startswith("ENPP_")
        source_missing = list(missing_rows(raw,name,"source",depth_optional))
        standard_missing = list(missing_rows(selected,name,"standard",depth_optional))
        missing.extend(source_missing+standard_missing)
        w=cover(name,scoped_raw,scoped_quality,selected,lo,hi,cut)
        rate.append(w)
        g=gaps(name,scoped_raw,selected,lo,hi,cut)
        largest_gaps.extend(g)
        diagnostics(name,scoped_raw,scoped_quality,selected,cfg,mc,curves,strata)
        integrity.append(audit)
        required_bad = sum(r["missing"]+r["invalid"] for r in standard_missing if r["required"])
        overview.append(dict(dataset=name,batch=batch,source_path=location,source_basis=basis,cutoff=cut,
            source_file_records=len(raw),source_records=len(scoped_raw),quality_records=len(scoped_quality),
            standard_records=len(selected),standard_missing_or_invalid_required_cells=required_bad,
            source_empty_bins=int(w.source.eq(0).sum()),standard_empty_bins=int(w.standard.eq(0).sum()),
            bin_frequency=w.frequency.iloc[0],longest_standard_gap_days=g[0]["days"],
            source_events_inside_longest_standard_gap=g[0]["source_inside"],
            first_source_event=str(scoped_raw.time.min()),last_source_event=str(scoped_raw.time.max()),
            first_standard_event=str(selected.time.min()),last_standard_event=str(selected.time.max())))
        print("DIAGNOSTICS", name, required_bad, w.source.eq(0).sum(), w.standard.eq(0).sum(), flush=True)
    for filename, rows in [("overview",overview),("missing_fields",missing),("coverage_gaps",largest_gaps),
                            ("mc_temporal",mc),("fmd",curves),("mc_depth",strata)]:
        pd.DataFrame(rows).to_csv(out/(filename+".csv"),index=False)
    pd.concat(rate,ignore_index=True).to_csv(out/"coverage_calendar.csv",index=False)
    (out/"cache_integrity.json").write_text(json.dumps(integrity,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("/root/autodl-tmp/em_eqf"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--dataset",action="append",default=[])
    args=parser.parse_args()
    run(args.root.resolve(),args.output,args.dataset)
