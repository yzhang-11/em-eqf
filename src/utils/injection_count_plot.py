"""Plot injection-rate and earthquake-count overlays for CIG datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.utils.cig_plotting import draw_split_markers, format_start_label, panel_label
from src.utils.injection_data import extract_injection_count_series
from src.utils.tpp_experiments import load_tpp_catalog


@dataclass(frozen=True)
class InjectionCountPlotConfig:
    start_mode: str = "val"
    # Okabe-Ito blue/orange provide robust contrast in print and for common
    # forms of colour-vision deficiency.
    injection_color: str = "#0072B2"
    count_color: str = "#D55E00"
    count_alpha: float = 0.25
    injection_linewidth: float = 1.35
    injection_label: str = r"Injection rate (m$^3$ min$^{-1}$)"
    daily_counts: bool = True
    count_bar_width_days: float = 0.90
    # Retain the original wide 2 x 2 layout used by this notebook.
    figsize: tuple[float, float] = (13.8, 5.6)
    ncols: int = 2
    title_fontsize: float = 10.5
    axis_label_fontsize: float = 9.5
    tick_labelsize: float = 8.5
    panel_label_fontsize: float = 10.0
    grid_alpha: float = 0.22
    split_label_y: float = 0.88


def load_injection_count_series(
    item: Mapping[str, Any],
    *,
    data_root: str | Path,
    fallback_catalog_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = str(item["dataset"])
    catalog, registry_name, _ = load_tpp_catalog(
        dataset,
        base_dir=Path(data_root) / dataset,
        catalog_cfg=item.get("catalog_cfg") or dict(fallback_catalog_config or {}),
        candidates=[f"{dataset}-Standard", dataset],
    )
    series = extract_injection_count_series(catalog.full_sequence, catalog.metadata)
    return {"series": series, "registry_name": registry_name, "metadata": catalog.metadata}


def crop_injection_count_series(
    series: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, np.ndarray | float]:
    start_days = float(item["plot_start_days"])
    end_days = float(item["curve_df"]["time_days"].max())
    absolute_days = np.asarray(series["absolute_days"], dtype=float)
    mask = (absolute_days >= start_days) & (absolute_days <= end_days)
    return {
        "x": absolute_days[mask] - start_days,
        "injection": np.asarray(series["injection"], dtype=float)[mask],
        "counts": np.asarray(series["counts"], dtype=float)[mask],
        "xmax": max(end_days - start_days, 1e-9),
    }


def daily_count_bars(x: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if x.size == 0:
        return np.array([]), np.array([])
    day_index = np.floor(np.maximum(x, 0.0)).astype(int)
    daily = pd.Series(counts).groupby(day_index).sum()
    return daily.index.to_numpy(dtype=float), daily.to_numpy(dtype=float)


def _set_data_ylim(ax, values: np.ndarray, headroom: float = 0.12) -> None:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    y_min = 0.0 if np.nanmin(values) >= 0 else float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    span = y_max - y_min
    padding = max(span * headroom, abs(y_max) * headroom, 1e-6)
    ax.set_ylim(y_min, y_max + padding)


def plot_injection_count_panel(
    ax,
    item: Mapping[str, Any],
    overlay_item: Mapping[str, Any],
    panel_index: int,
    *,
    config: InjectionCountPlotConfig,
):
    cropped = crop_injection_count_series(overlay_item["series"], item)
    x = np.asarray(cropped["x"])
    injection = np.asarray(cropped["injection"])
    counts = np.asarray(cropped["counts"])
    count_axis = ax.twinx()
    ax.set_zorder(count_axis.get_zorder() + 1)
    ax.patch.set_visible(False)

    if config.daily_counts:
        count_x, count_y = daily_count_bars(x, counts)
        # ``daily_count_bars`` returns the start of each day.  Centering the
        # rectangle on the day makes the temporal support unambiguous.
        count_axis.bar(
            count_x + 0.5,
            count_y,
            width=config.count_bar_width_days,
            align="center",
            color=config.count_color,
            edgecolor="none",
            alpha=config.count_alpha,
            label="Daily event count",
            zorder=1,
        )
        count_values = count_y
    else:
        nonzero = counts > 0
        count_axis.vlines(
            x[nonzero], 0, counts[nonzero], color=config.count_color,
            alpha=0.72, linewidth=0.7, label="Event count",
        )
        count_values = counts[nonzero]

    ax.plot(
        x,
        injection,
        color=config.injection_color,
        linewidth=config.injection_linewidth,
        label=config.injection_label,
        zorder=3,
    )
    ax.set_xlim(0, cropped["xmax"])
    _set_data_ylim(ax, injection)
    if count_values.size:
        _set_data_ylim(count_axis, count_values)
        count_axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    ax.set_title(item["title"], fontsize=config.title_fontsize, pad=5)
    ax.text(
        0.01, 0.99, panel_label(panel_index), transform=ax.transAxes,
        va="top", ha="left", fontsize=config.panel_label_fontsize,
        fontweight="bold",
    )
    ax.set_xlabel(
        f"Days since {format_start_label(config.start_mode)}",
        fontsize=config.axis_label_fontsize,
        labelpad=3,
    )
    ax.set_ylabel(config.injection_label, color=config.injection_color, fontsize=config.axis_label_fontsize)
    count_axis.set_ylabel(
        "Event count / day" if config.daily_counts else "Event count",
        color=config.count_color, fontsize=config.axis_label_fontsize,
        labelpad=5,
    )
    ax.tick_params(axis="y", labelcolor=config.injection_color, labelsize=config.tick_labelsize, length=3)
    count_axis.tick_params(axis="y", labelcolor=config.count_color, labelsize=config.tick_labelsize, length=3)
    ax.tick_params(axis="x", labelsize=config.tick_labelsize, length=3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True, min_n_ticks=3))
    ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=config.grid_alpha, zorder=0)
    ax.spines["top"].set_visible(False)
    count_axis.spines["top"].set_visible(False)
    count_axis.spines["right"].set_linewidth(0.8)
    draw_split_markers(ax, item, label_y=config.split_label_y)
    return count_axis


def plot_injection_count_grid(
    items: Sequence[Mapping[str, Any]],
    *,
    data_root: str | Path,
    config: InjectionCountPlotConfig,
    fallback_catalog_config: Mapping[str, Any] | None = None,
) -> tuple[Any, np.ndarray, dict[str, dict[str, Any]]]:
    overlays: dict[str, dict[str, Any]] = {}
    for item in items:
        overlay = load_injection_count_series(
            item, data_root=data_root, fallback_catalog_config=fallback_catalog_config
        )
        overlays[str(item["dataset"])] = overlay
        print(
            f"Loaded {item['dataset']} injection/count series from {overlay['registry_name']} "
            f"({overlay['series']['source']})."
        )

    nrows = int(np.ceil(len(items) / config.ncols))
    fig, axes = plt.subplots(nrows, config.ncols, figsize=config.figsize, squeeze=False)
    flat_axes = axes.ravel()
    for index, (ax, item) in enumerate(zip(flat_axes, items)):
        plot_injection_count_panel(
            ax, item, overlays[str(item["dataset"])], index, config=config
        )
    for ax in flat_axes[len(items):]:
        ax.axis("off")

    fig.legend(
        handles=[
            Line2D(
                [0], [0], color=config.injection_color,
                linewidth=config.injection_linewidth, label=config.injection_label,
            ),
            Patch(
                facecolor=config.count_color,
                edgecolor=config.count_color,
                alpha=config.count_alpha,
                label="Daily event count" if config.daily_counts else "Event count",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        fontsize=8.5,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.2,
        handletextpad=0.55,
    )
    fig.tight_layout(rect=(0.03, 0.10, 0.97, 0.98), w_pad=1.0, h_pad=1.1)
    return fig, axes, overlays
