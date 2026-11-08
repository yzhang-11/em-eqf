"""Plot frequency–magnitude and coverage diagnostics from the audit CSV files."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RECENT = ["Taiwan_CWA", "Taiwan_Wu2008", "Turkiye_EAFZ", "Chile_Atacama",
          "Chile_Potin", "Greece_WestHellenic", "Greece_SantoriniSwarm"]
COLORS = dict(source="#848a92", quality="#247493", standard="#b64f34")


def plot(path):
    overview = pd.read_csv(path/"overview.csv").set_index("dataset")
    fmd = pd.read_csv(path/"fmd.csv")
    fig, grid = plt.subplots(4, 2, figsize=(11, 11), layout="constrained")
    for axis, name in zip(grid.flat, RECENT):
        data = fmd[fmd.dataset.eq(name)]
        for stage, color in COLORS.items():
            part = data[data.stage.eq(stage)]
            axis.plot(part.magnitude_edge, part["count"].where(part["count"].gt(0)),
                      color=color, label=stage, linewidth=1.4)
        cut = overview.loc[name, "cutoff"]
        axis.axvline(cut, color="#222222", linestyle="--", linewidth=1)
        axis.set(title=name + f"  |  cutoff {cut:g}", yscale="log", xlabel="Magnitude bin lower edge", ylabel="Events / 0.1 bin")
        axis.grid(alpha=.2)
    grid.flat[-1].axis("off")
    handles, labels = grid.flat[0].get_legend_handles_labels()
    grid.flat[-1].legend(handles, labels, loc="center", frameon=False, title="Catalog stage")
    fig.suptitle("Frequency–magnitude audit | original scales, separate catalogs")
    fig.savefig(path/"recent_magnitude_distributions.png", dpi=175)
    plt.close(fig)

    bins = pd.read_csv(path/"coverage_calendar.csv")
    original_types = path/"ncedc_2024_raw_types.csv"
    fig, grid = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for axis, name in zip(grid.flat, ["NCEDC","USGS_Montana","Taiwan_CWA","Chile_Potin"]):
        part = bins[bins.dataset.eq(name)]
        x = pd.to_datetime(part.start, utc=True, format="mixed")
        for stage in ("source", "standard"):
            label = "parsed reference" if stage == "source" else "standard"
            axis.plot(x,part[stage],label=label,color=COLORS[stage],linewidth=1.25)
        if name == "NCEDC" and original_types.exists():
            raw = pd.read_csv(original_types, index_col="month")
            axis.plot(pd.to_datetime(raw.index, utc=True), raw.sum(axis=1), color="#247493",
                      linewidth=2, linestyle="--", label="2024 original archive, all types")
        empty = part.standard.eq(0)
        axis.scatter(x[empty],np.zeros(int(empty.sum())),s=15,c="#b64f34",
                     label="empty standard bin",zorder=4)
        axis.set(title=name,yscale="symlog",ylabel="Events per calendar month")
        locator = mdates.AutoDateLocator(minticks=4,maxticks=7)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.grid(alpha=.2)
    grid.flat[0].legend(frameon=False,loc="lower left",fontsize=8)
    fig.suptitle("Coverage review | zero records marked, source and screened series aligned")
    fig.savefig(path/"coverage_diagnostics.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shared/reports",type=Path)
    plot(parser.parse_args().reports)
