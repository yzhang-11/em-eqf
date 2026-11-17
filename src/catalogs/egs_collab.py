"""Five separately observed stimulation sequences at the same SURF testbed."""
import json
from pathlib import Path

import pandas as pd
import torch

from src.data import Catalog, default_catalogs_dir
from src.data.preprocessing import estimate_mc_max_curvature
from .induced_triplet_base import InducedTripletBase


class EGSCollabStage(InducedTripletBase):
    @classmethod
    def resolve_site_data_dir(cls, path):
        # Older configs use data/<stage>. New releases share a single site directory.
        if path.name == cls.stage and not (path / "processed").is_dir():
            return path.parent / "EGSCollab_E1"
        return path

    def __init__(self, root_dir=None, data_dir=None, mag_completeness=None, normalize=False,
                 train_start_ts=None, val_start_ts=None, test_start_ts=None, **kwargs):
        name = self.stage
        data_dir = self.resolve_site_data_dir(Path(data_dir)) if data_dir else default_catalogs_dir / "EGSCollab_E1"
        root_dir = Path(root_dir) if root_dir else data_dir / "catalogs"
        prefix = data_dir / "processed"
        summary = json.loads((prefix / f"{name}_summary.json").read_text())
        val = val_start_ts if val_start_ts is not None else summary["val_start_ts"]
        test = test_start_ts if test_start_ts is not None else summary["test_start_ts"]
        if mag_completeness is None and (train_start_ts is not None or val_start_ts is not None or kwargs.get("end_ts") is not None):
            events = pd.read_csv(prefix / f"{name}_eq_processed.csv")
            time = pd.to_datetime(events.time_iso, utc=True)
            start = pd.to_datetime(train_start_ts or summary["start_time_iso"], utc=True)
            stop = min(pd.to_datetime(val, utc=True),
                       pd.to_datetime(kwargs.get("end_ts") or summary["end_time_iso"], utc=True))
            mag_completeness = round(float(estimate_mc_max_curvature(
                events.loc[(time >= start) & (time < stop), "magnitude"], plot=False)[0]), 2)
        super().__init__(name, root_dir, data_dir=data_dir, normalize=normalize,
                         mag_completeness=mag_completeness, train_start_ts=train_start_ts,
                         val_start_ts=val, test_start_ts=test, **kwargs)

    def _prepare_inj_dataframe(self):
        # The native table is a complete clock-minute grid of backward-looking means.
        source = pd.read_csv(self.inj_file)
        t = self._relative_time_from_columns(source, kind="inj")
        frame = pd.DataFrame({"t": t, "raw_rate": source.inj_rate_m3_min})
        frame = frame.loc[(t >= 0) & (t <= self.t_end)].iloc[::self.freq_min].copy()
        if len(frame) < 2:
            raise ValueError("Requested interval contains fewer than two pump measurements")
        frame["rate"] = frame.raw_rate
        if self.normalize_time_series:
            train = frame.loc[(frame.t >= self.metadata["train_start_t"]) &
                              (frame.t < self.metadata["val_start_t"]), "raw_rate"]
            if train.empty:
                raise ValueError("No training flow readings for normalization")
            lo, hi = float(train.min()), float(train.max())
            frame["rate"] = (frame.raw_rate - lo) / (hi - lo if hi > lo else 1e-8)
            self.norm_stats["inj_rate"] = {"min": lo, "max": hi}
            torch.save(self.norm_stats, self.root_dir / "norm_stats.pt")
        return frame


for _episode in ("522", "523", "524", "525A", "525B"):
    _name = f"EGSCollab_{_episode}"
    _stage = type(_name, (EGSCollabStage,), {"stage": _name, "__module__": __name__})
    globals()[_name] = Catalog.register(name=_name + "-Standard")(_stage)
