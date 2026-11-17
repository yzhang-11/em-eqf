"""Build the five May 2018 N164 stimulation triplets from pinned GDR inputs."""
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.preprocessing import estimate_mc_max_curvature

SITE = ROOT / "data/EGSCollab_E1"
RAW = SITE / "raw"
PUMPS = {
    "522": ("SF_E1_Stim2_1.5m_May22.csv", "Quizix Flow", "22", 0),
    "523": ("SF_E1_Stim2_1.5mStepTest.csv", "Quizix Flow", "23", 0),
    "524": ("SF_E1_Stim2_DriveToProduction.csv", "Triplex Net Flow", "24", 1),
    "525A": ("SF_E1_Stim2_DriveToProduction.csv", "Triplex Net Flow", "25", 0),
    "525B": ("SF_E1_Stim2_DriveToProduction.csv", "Triplex Net Flow", "25", 1),
}


def read_pump(archive, member, channel):
    with archive.open(member) as stream:
        stream.readline()  # Export header; recording times are in the individual rows.
        columns = stream.readline().decode().strip().split(",")
        units = stream.readline().decode().strip().split(",")
        i = columns.index(channel)
        if columns[0] != "Time" or units[i] != "LPM":
            raise ValueError(f"Unexpected pump schema: {member}")
        data = pd.read_csv(stream, header=None, usecols=[0, i], encoding="latin1")
    time = pd.to_datetime(data[0], format="%m/%d/%y %H:%M:%S", utc=True)
    rate = pd.Series(data[i].to_numpy(float), index=time)
    if not np.isfinite(rate).all() or not time.is_monotonic_increasing:
        raise ValueError(f"Invalid flow readings: {member}")
    return rate.groupby(level=0).mean()


def waveform_blocks(inventory, day):
    prefix = f"egs_collab/experiment_1/microseismic_monitoring/201805{day}/"
    starts = pd.Series(pd.to_datetime(
        [s.removeprefix("vbox_").removesuffix(".dat") for s in inventory[prefix]],
        format="%Y%m%d%H%M%S%f", utc=True,
    )).sort_values()
    # Typical file starts are 34 s apart. Split when a full file could be missing.
    groups = (starts.diff() > pd.Timedelta(seconds=50)).cumsum()
    return [(g.iloc[0], g.iloc[-1]) for _, g in starts.groupby(groups)]


def build_triplet(label, catalog, inventory, rate):
    member, channel, day, block = PUMPS[label]
    first, last = waveform_blocks(inventory, day)[block]
    minute = pd.Timedelta(minutes=1)
    start = max(first, rate.index.min() + minute).ceil("min")
    end = min(last, rate.index.max()).floor("min")
    grid = pd.date_range(start, end, freq="min")
    # Bin (t-60s, t], timestamp the completed measurement at t.
    bins = rate.resample("min", closed="right", label="right").agg(["mean", "count"]).reindex(grid)
    if len(grid) < 2 or bins.isna().any().any() or bins["count"].min() < 50:
        raise ValueError(f"Insufficient pump coverage in {label}")
    inj = pd.DataFrame({"time_iso": grid, "inj_rate_m3_min": bins["mean"].values / 1000,
                        "observed_seconds": bins["count"].values})
    eq = catalog.loc[catalog["time_iso"].between(start, end, inclusive="left")].copy()
    eq = eq.loc[eq.magnitude.notna()].sort_values("time_iso").reset_index(drop=True)
    time = eq.time_iso
    def boundary(frac):
        i = int(len(time) * frac)
        return time[i-1] + (time[i] - time[i-1]) / 2
    val, test = boundary(.70), boundary(.85)
    mc = round(float(estimate_mc_max_curvature(eq.loc[time < val, "magnitude"],
                                               bin_width=.1, plot=False)[0]), 2)
    keep = eq.magnitude > mc  # InducedTripletBase uses a strict magnitude cutoff.
    split_counts = [int((keep & mask).sum()) for mask in
                    (time < val, (time >= val) & (time < test), time >= test)]
    name = f"EGSCollab_{label}"
    dest = ROOT / "data" / name / "processed"
    dest.mkdir(parents=True, exist_ok=True)
    eq.to_csv(dest / f"{name}_eq_processed.csv", index=False, lineterminator="\n")
    inj.to_csv(dest / f"{name}_inj_1min_processed.csv", index=False, lineterminator="\n")
    summary = dict(dataset=name, site="EGS Collab Experiment 1 / N164", protocol="v1",
                   observation="intersection of published waveform file-start span and pump coverage",
                   source_member=member, source_channel=channel, magnitude_type="mag_rel",
                   mc=mc, mc_method="training-prefix MAXC, 0.1 bins, exploratory; retain M > Mc",
                   resample_freq_min=1, is_upsample=False, inj_fill_policy="past_minute_observed_mean",
                   start_time_iso=start.isoformat(), end_time_iso=end.isoformat(),
                   val_start_ts=val.isoformat(), test_start_ts=test.isoformat(),
                   n_eq=len(eq), n_scored=split_counts, n_minute_samples=len(inj),
                   min_observed_seconds=int(bins["count"].min()),
                   peak_minute_l_min=float(bins["mean"].max()),
                   event_id_column="eventid", splits="chronological 70/15/15 before Mc",
                   column_aliases={"eq_time":"time_iso","inj_time":"time_iso"})
    (dest / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, set(eq.eventid)


def main():
    pins = json.loads((SITE / "sources.json").read_text())
    for name, expected in pins["files"].items():
        with (RAW / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"Source checksum mismatch: {name}")
    catalog = pd.read_csv(RAW / "catalog_manual_all_v200629.csv").rename(columns={"mag_rel":"magnitude"})
    catalog["time_iso"] = pd.to_datetime(catalog.date, utc=True, format="ISO8601")
    if catalog.eventid.duplicated().any() or catalog.time_iso.duplicated().any():
        raise ValueError("Duplicate source identity or origin time")
    inventory = json.loads((RAW / "waveform_inventory.json").read_text())
    seen = set()
    with ZipFile(RAW / "may_pump_records.zip") as archive:
        rates = {}
        for label, (member, channel, _, _) in PUMPS.items():
            if member not in rates:
                rates[member] = read_pump(archive, member, channel)
            summary, ids = build_triplet(label, catalog, inventory, rates[member])
            if ids & seen:
                raise ValueError("Event reused between stimulation windows")
            seen.update(ids)
            print(label, summary["n_eq"], summary["mc"], summary["n_scored"])


if __name__ == "__main__":
    main()
