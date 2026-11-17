"""Integration checks for the five real GDR catalog/pump pairs (requires LFS data)."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import default_catalogs_dir
from src.data.preparation_modules.tpp import prepare_data_tpp


@pytest.mark.parametrize("episode", ["522", "523", "524", "525A", "525B"])
def test_scored_events_and_flow(episode, tmp_path):
    name = f"EGSCollab_{episode}"
    data_dir = default_catalogs_dir / name
    prefix = data_dir / "processed"
    meta = json.loads((prefix / f"{name}_summary.json").read_text())
    events = pd.read_csv(prefix / f"{name}_eq_processed.csv")
    events = events.loc[events.magnitude > meta["mc"]]
    times = pd.to_datetime(events.time_iso, utc=True)
    start = pd.Timestamp(meta["start_time_iso"])
    inj = pd.read_csv(prefix / f"{name}_inj_1min_processed.csv")
    args = SimpleNamespace(dataset=name, batch_size=1,
                           catalog_cfg={"root_dir": tmp_path / "catalogs", "normalize": True})
    seq, train, val, test, catalog = prepare_data_tpp(args, data_dir)
    assert [args.num_events_train, args.num_events_val, args.num_events_test] == meta["n_scored"]
    for loader in (train, val, test):
        next(iter(loader))
    cut1, cut2 = [pd.Timestamp(meta[key]) for key in ("val_start_ts", "test_start_ts")]
    for subset, mask in zip((catalog.train, catalog.val, catalog.test),
                            (times < cut1, (times >= cut1) & (times < cut2), times >= cut2)):
        sample = subset[0]
        scored = sample.arrival_times >= sample.t_nll_start
        hours = (times[mask] - start).dt.total_seconds() / 3600
        np.testing.assert_allclose(sample.arrival_times[scored], hours, rtol=0, atol=4e-7)
        np.testing.assert_allclose(sample.mag[scored], events.loc[mask, "magnitude"])
    assert (seq.inter_times > 0).all()
    itime = pd.to_datetime(inj.time_iso, utc=True)
    fit = inj.loc[itime < cut1, "inj_rate_m3_min"]
    expected = (inj.inj_rate_m3_min - fit.min()) / (fit.max() - fit.min())
    np.testing.assert_allclose(seq.time_series[:, 0], expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(seq.raw_time_series[:, 0], inj.inj_rate_m3_min, atol=1e-9)
