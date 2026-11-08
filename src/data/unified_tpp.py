"""Shared input for models evaluated on a unified TPP catalog."""
from pathlib import Path

import numpy as np
import torch

from .batch import Batch
from .sequence import Sequence

NS_PER_DAY = 86_400_000_000_000


def load_catalogs(path: Path, *, test: bool = False):
    """Open only development events unless final test events are requested."""
    return torch.load(Path(path) / ("test.pt" if test else "development.pt"),
                      map_location="cpu", weights_only=False)


def calendar_batch(region, start_ns: int, end_ns: int, *, history: int | None = None,
                   region_id: int = 0) -> Batch:
    """Read [start_ns, end_ns) with an optional cap on preceding event context.

    Time input is float64 in days. The final interval holds the time from the
    last event to end_ns. Only the target events enter nll_event_mask.
    """
    ns = region["ns"]
    if not region["time_bounds_ns"]["start"] <= start_ns < end_ns <= region["time_bounds_ns"]["test_end"]:
        raise ValueError("Requested interval is outside the catalog observation period")
    if "test_end" not in region and end_ns > region["time_bounds_ns"]["val_end"]:
        raise ValueError("Development catalog ends before the requested interval")
    if history is not None and history < 1:
        raise ValueError("History must be positive")

    first, end = map(int, np.searchsorted(ns, [start_ns, end_ns]))
    left = 0 if history is None else max(0, first - history)
    if end == left:
        raise ValueError("Window has no observed events or preceding history")
    origin = int(ns[left - 1]) if left else region["time_bounds_ns"]["start"]
    times = np.concatenate(([origin], ns[left:end], [end_ns]))
    nll_start = (start_ns - origin) / NS_PER_DAY
    seq = Sequence(torch.from_numpy(np.diff(times) / NS_PER_DAY),
                   t_nll_start=nll_start,
                   mag=torch.from_numpy(region["magnitude"][left:end].copy()),
                   event_index=torch.arange(left, end))
    seq.region_params = torch.as_tensor(region["metadata"], dtype=torch.float64).expand(end - left, -1).clone()
    if "mag_mean_train" in region:
        seq.region_mag_mean = torch.full((end - left, 1), region["mag_mean_train"], dtype=torch.float64)
    seq.region_id = torch.full((end - left,), region_id, dtype=torch.long)
    batch = Batch.from_list([seq])
    # Batch defaults to a left-open interval. Select [start, end) exactly by
    # integer catalog indices, including any event precisely at start_ns.
    batch.start_idx[0] = first - left
    batch.nll_event_mask.zero_()
    batch.nll_event_mask[0, first - left:end - left] = 1
    return batch
