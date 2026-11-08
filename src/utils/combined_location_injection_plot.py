"""Combined study-location map and injection/count time-series figure."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.utils.cig_plotting import panel_label
from src.utils.dataset_location_map import (
    LocationMapConfig,
    _plot_world_background,
    build_location_items,
)
from src.utils.injection_count_plot import (
    InjectionCountPlotConfig,
    load_injection_count_series,
    plot_injection_count_panel,
)


def _location_groups(locations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the two Cooper Basin directories into one labeled site."""

    cooper = [
        location
        for location in locations
        if str(location["dataset"]) in {"CB_HAB1a", "CB_HAB4"}
    ]
    groups: list[dict[str, Any]] = []
    if cooper:
        groups.append(
            {
                "label": "Cooper Basin: CB1a, CB4",
                "lon": float(np.mean([location["center_lon"] for location in cooper])),
                "lat": float(np.mean([location["center_lat"] for location in cooper])),
            }
        )

    cooper_datasets = {str(location["dataset"]) for location in cooper}
    for location in locations:
        if str(location["dataset"]) in cooper_datasets:
            continue
        groups.append(
            {
                "label": str(location["title"]),
                "lon": float(location["center_lon"]),
                "lat": float(location["center_lat"]),
            }
        )
    return groups


def _draw_compact_location_map(
    axis,
    loaded_items: Sequence[Mapping[str, Any]],
    config: LocationMapConfig,
    *,
    map_region: Sequence[float],
) -> None:
    """Draw a restrained world backdrop and directly labeled study sites."""

    background_source = _plot_world_background(axis, config)
    west, east, south, north = (float(value) for value in map_region)
    # Preserve the lon/lat geometry of the equirectangular background. The
    # compact top row otherwise makes continents visibly too wide.
    axis.set_box_aspect((north - south) / (east - west))
    axis.set_xticks(np.arange(np.ceil(west / 60) * 60, np.floor(east / 60) * 60 + 1, 60))
    axis.set_yticks(np.arange(np.ceil(south / 30) * 30, np.floor(north / 30) * 30 + 1, 30))
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.tick_params(axis="both", labelsize=6.5, pad=1)
    axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.28, zorder=1)

    locations = build_location_items(loaded_items, config)
    for group in _location_groups(locations):
        axis.scatter(
            group["lon"],
            group["lat"],
            s=66,
            marker="*",
            color="#202020",
            edgecolor="white",
            linewidth=0.75,
            zorder=4,
        )
        # Keep labels in a single visual language; map marker colors do not
        # reuse the blue/orange meaning reserved for the time-series panel.
        if group["label"].startswith("Cooper Basin"):
            offset = (-5.0, 5.0)
            horizontal_alignment = "right"
            vertical_alignment = "bottom"
        elif group["label"] == "ST1-2018":
            offset = (5.0, -4.0)
            horizontal_alignment = "left"
            vertical_alignment = "top"
        else:
            offset = (6.0, 4.5)
            horizontal_alignment = "left"
            vertical_alignment = "bottom"
        axis.annotate(
            group["label"],
            xy=(group["lon"], group["lat"]),
            xytext=(group["lon"] + offset[0], group["lat"] + offset[1]),
            fontsize=8,
            color="#202020",
            ha=horizontal_alignment,
            va=vertical_alignment,
            arrowprops={
                "arrowstyle": "-",
                "color": "#202020",
                "linewidth": 0.6,
                "shrinkA": 0,
                "shrinkB": 2,
            },
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.25},
            zorder=5,
        )

    # These quiet geographic cues make the two relevant parts of the map easy
    # to find without adding a second color encoding for the catalogs.
    axis.text(
        0.50,
        0.83,
        "Europe",
        transform=axis.transAxes,
        fontsize=8,
        color="#606060",
        fontstyle="italic",
        ha="center",
    )
    axis.text(
        0.87,
        0.16,
        "Australia",
        transform=axis.transAxes,
        fontsize=8,
        color="#606060",
        fontstyle="italic",
        ha="center",
    )
    axis.text(
        0.995,
        0.015,
        f"Background: {background_source}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#6a6a6a",
    )
    axis.text(
        0.01,
        0.98,
        panel_label(0),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#222222",
    )
    axis.set_title("Study locations", fontsize=9.5, pad=2)


def plot_location_injection_count_combined(
    loaded_items: Sequence[Mapping[str, Any]],
    *,
    data_root: str | Path,
    location_config: LocationMapConfig,
    injection_config: InjectionCountPlotConfig,
    figsize: tuple[float, float] = (13.8, 7.0),
    map_width_fraction: float = 0.68,
    map_region: Sequence[float] = (-180, 180, -45, 65),
) -> tuple[Any, np.ndarray, dict[str, dict[str, Any]]]:
    """Return a centered map over a full-width 2 x 2 injection/count grid.

    The map deliberately uses a single dark marker style.  The lower panels
    retain :func:`plot_injection_count_panel`, including their twin y-axes and
    the blue injection/orange event-count convention.
    """

    if not 0.5 <= float(map_width_fraction) <= 0.85:
        raise ValueError("map_width_fraction should be between 0.5 and 0.85.")
    if len(loaded_items) == 0:
        raise ValueError("At least one loaded dataset is required.")

    data_root = Path(data_root)
    fig = plt.figure(figsize=figsize)
    figure_left, figure_right = 0.055, 0.965
    figure_width = figure_right - figure_left
    # Reserve at most the requested fraction of the complete figure width for
    # the map. ``set_box_aspect`` below may make the actual map narrower so
    # that longitude/latitude geometry remains undistorted.
    map_grid_fraction = float(map_width_fraction) / figure_width
    if map_grid_fraction >= 0.98:
        raise ValueError("map_width_fraction leaves too little side margin.")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.35, 2.15),
        hspace=0.12,
    )
    # The side columns center the map slot over the full-width time series.
    map_grid = outer[0].subgridspec(
        1,
        3,
        width_ratios=(1.0, 2.0 * map_grid_fraction / (1.0 - map_grid_fraction), 1.0),
        wspace=0.0,
    )
    map_axis = fig.add_subplot(map_grid[0, 1])
    compact_map_config = replace(location_config, show_location_points=False)
    _draw_compact_location_map(
        map_axis,
        loaded_items,
        compact_map_config,
        map_region=map_region,
    )

    ncols = int(injection_config.ncols)
    nrows = int(np.ceil(len(loaded_items) / ncols))
    time_grid = outer[1].subgridspec(nrows, ncols, wspace=0.28, hspace=0.65)
    axes = np.empty((nrows, ncols), dtype=object)
    overlays: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(loaded_items):
        row, column = divmod(index, ncols)
        axis = fig.add_subplot(time_grid[row, column])
        axes[row, column] = axis
        overlay = load_injection_count_series(item, data_root=data_root)
        overlays[str(item["dataset"])] = overlay
        plot_injection_count_panel(
            axis,
            item,
            overlay,
            index + 1,
            config=injection_config,
        )
        print(
            f"Loaded {item['dataset']} injection/count series from {overlay['registry_name']} "
            f"({overlay['series']['source']})."
        )
    for index in range(len(loaded_items), nrows * ncols):
        row, column = divmod(index, ncols)
        axes[row, column] = fig.add_subplot(time_grid[row, column])
        axes[row, column].axis("off")

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=injection_config.injection_color,
                linewidth=injection_config.injection_linewidth,
                label=injection_config.injection_label,
            ),
            Patch(
                facecolor=injection_config.count_color,
                edgecolor=injection_config.count_color,
                alpha=injection_config.count_alpha,
                label="Daily event count" if injection_config.daily_counts else "Event count",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        fontsize=8.5,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.2,
        handletextpad=0.55,
    )
    fig.subplots_adjust(left=figure_left, right=figure_right, top=0.965, bottom=0.085)
    return fig, axes, overlays
