"""Rebuild the Bedretto VALTER paired catalog from the author's pinned archive.

One site, not independent regions: all experiments share stimulation borehole ST1.
Requires numpy, pandas and h5py. Does not interpolate across experiment gaps.
"""
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


def build():
    site = DATA / "Bedretto_VALTER"
    manifest = json.loads((site / "sources.json").read_text(encoding="utf8"))
    raw = site / "raw" / manifest["files"][0]["file"]
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == manifest["files"][0]["sha256"]
    with ZipFile(raw) as archive:
        def nested(suffix):
            member, = [n for n in archive.namelist() if n.endswith(suffix)]
            return ZipFile(io.BytesIO(archive.read(member)))
        catalogs = nested("data/BULGGHighQualitySeismicCatalogsVel5400.zip")
        hydraulics = nested("data/HydroData.zip")
    events, reports, windows, coverage = [], [], [], []
    for member in sorted(catalogs.namelist()):
        if not Path(member).name.startswith("Catalog_") or "__MACOSX" in member:
            continue
        experiment = re.search(r"Catalog_(.*?)_HypoDD", member)[1]
        hydro_name = "HydroData_" + experiment.replace("9_10", "9+10") + ".h5"
        hydro_member, = [n for n in hydraulics.namelist()
                        if Path(n).name == hydro_name and "__MACOSX" not in n]
        with h5py.File(io.BytesIO(hydraulics.read(hydro_member)), "r") as h:
            group = h["df"]
            frame = pd.concat([pd.DataFrame(group[f"block{i}_values"][:],
                                 columns=[s.decode() for s in group[f"block{i}_items"][:]])
                               for i in range(int(group.attrs["nblocks"]))], axis=1)
            if group["axis1"].attrs["kind"] == b"datetime64":
                clock_values = group["axis1"][:]
            else:
                # Later experiments store UTC epoch nanoseconds in 'time'; the
                # HDF index is merely a row number, not an epoch timestamp.
                clock_values = frame.time.to_numpy()
                assert np.min(clock_values) > 1e18
            time = pd.to_datetime(clock_values, unit="ns", utc=True)
        frame = frame.rename(columns=lambda name: name.removesuffix(".value"))
        inj = pd.DataFrame({"time_iso": time, "hydraulic_record": experiment,
                            "well_id": "Bedretto:ST1", "net_flow_m3_s": frame.topflow,
                            "inj_rate_m3_min": frame.topflow.where(frame.topflow >= 0) * 60,
                            "wellhead_pressure_pa": frame.get("toppressure", np.nan),
                            "bottom_pressure_pa": frame.get("bottompressure", np.nan)})
        duplicate_observations = inj.time_iso.duplicated(keep=False)
        inj = inj.loc[~duplicate_observations].sort_values("time_iso").reset_index(drop=True)
        # Retain native measurements; negative flow is flowback, not water injection.
        source = pd.read_csv(io.BytesIO(catalogs.read(member)))
        eq = pd.DataFrame({"event_id": experiment + ":row:" + (source.index + 2).astype(str),
                           "time_iso": pd.to_datetime(source.Time, utc=True),
                           "magnitude": source.MwA, "magnitude_type": "MwA",
                           "latitude": source.Latitude, "longitude": source.Longitude,
                           "source_depth_m": source.Depth, "hydraulic_record": experiment})
        eq = eq.sort_values("time_iso")
        # Coverage requires consecutive valid samples at most one native minute
        # apart. Never score across an unobserved gap or extend the final sample.
        clock = inj.time_iso.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        target = eq.time_iso.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        previous = np.searchsorted(clock, target, side="right") - 1
        valid = (previous >= 0) & (target <= clock[-1])
        idx = previous.clip(0)
        valid &= (target - clock[idx] <= 60_000_000_000)
        finite = np.isfinite(inj.inj_rate_m3_min.to_numpy())
        spans = finite[:-1] & finite[1:] & (np.diff(clock) <= 60_000_000_000)
        valid &= np.r_[spans, False][idx] & np.isfinite(eq.magnitude)
        starts = np.flatnonzero(spans & ~np.r_[False, spans[:-1]])
        stops = np.flatnonzero(spans & ~np.r_[spans[1:], False]) + 1
        for start, stop in zip(starts, stops):
            coverage.append(dict(hydraulic_record=experiment,
                                 start=inj.time_iso.iloc[start].isoformat(),
                                 end=inj.time_iso.iloc[stop].isoformat()))
        events.append(eq.loc[valid])
        reports.append(inj)
        windows.append(dict(hydraulic_record=experiment, source_events=len(eq),
                            paired_events=int(valid.sum()),
                            duplicate_observation_rows_excluded=int(duplicate_observations.sum()),
                            first_observation=inj.time_iso.iloc[0].isoformat(),
                            last_observation=inj.time_iso.iloc[-1].isoformat()))
    eq = pd.concat(events).sort_values("time_iso").reset_index(drop=True)
    tied = eq.time_iso.duplicated(keep=False)
    eq = eq.loc[~tied].reset_index(drop=True)
    inj = pd.concat(reports).sort_values("time_iso").reset_index(drop=True)
    assert not inj.time_iso.duplicated().any() and not eq.event_id.duplicated().any()
    cuts = []
    for fraction in [.7, .85]:
        i = int(len(eq) * fraction)
        cuts.append(eq.time_iso.iloc[i - 1] + (eq.time_iso.iloc[i] - eq.time_iso.iloc[i - 1]) / 2)
    a = eq.loc[eq.time_iso < cuts[0], "magnitude"].to_numpy()
    bins = np.arange(np.floor(a.min() * 10) / 10, np.ceil(a.max() * 10) / 10 + .1, .1)
    hist, edges = np.histogram(a, bins)
    i = hist.argmax()
    mc = round(float((edges[i] + edges[i + 1]) / 2), 2)
    scored = eq.loc[eq.magnitude > mc, "time_iso"]
    counts = [int((scored < cuts[0]).sum()), int(((scored >= cuts[0]) & (scored < cuts[1])).sum()),
              int((scored >= cuts[1]).sum())]
    assert counts[0] > 1500
    summary = dict(dataset="Bedretto_VALTER", events=len(eq), with_magnitude=len(eq),
                   injection_rows=len(inj), mc=mc, n_scored=counts,
                   val_start_ts=cuts[0].isoformat(), test_start_ts=cuts[1].isoformat(),
                   first_event=eq.time_iso.min().isoformat(), last_event=eq.time_iso.max().isoformat(),
                   mc_method="training-prefix MAXC, bin width 0.1, retain M > Mc",
                   tied_time_rows_excluded=int(tied.sum()), hydraulic_windows=windows,
                   negative_flow_rows=int((inj.net_flow_m3_s < 0).sum()),
                   observation_rule="Native UTC minute samples; no interpolation across gaps; see hydraulic_windows",
                   independent_regions=1, wells_with_valid_reports=1)
    out = site / "processed"
    out.mkdir(parents=True, exist_ok=True)
    eq.to_csv(out / "Bedretto_VALTER_eq_processed.csv", index=False)
    inj.to_csv(out / "Bedretto_VALTER_inj_processed.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "Bedretto_VALTER_observation_windows.csv", index=False)
    (out / "Bedretto_VALTER_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: summary[k] for k in ["events", "mc", "n_scored", "injection_rows"]}))


if __name__ == "__main__":
    build()
