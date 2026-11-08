from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import event_pipeline
from src.data.preparation_modules.lstm import _load_lstm_event_windows_with_cache
from src.utils import file_utils


@pytest.fixture
def catalog_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    t = np.arange(1., 41.)
    pd.DataFrame({"t": t, "Magnitude": 3 + t % 3,
                  "Latitude": t, "Longitude": t, "Depth": t,
                  "dt": np.ones(len(t))}).to_csv(raw / "events.csv", index=False)
    return tmp_path


def options():
    return dict(Mc=3, Mf=4, Twindow=4, Tfore=2, dt=2, context_len=0)


def test_tasks_and_feature_settings_share_windows(catalog_dir, monkeypatch):
    original = event_pipeline.event_loader.construct_samples_list
    calls = []

    def construct(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(event_pipeline.event_loader, "construct_samples_list", construct)
    a = event_pipeline.load_event_windows_for_task(catalog_dir, task_type="classification", **options())
    b = event_pipeline.load_event_windows_for_task(catalog_dir, task_type="regression", **options())
    args = SimpleNamespace(**options(), time_step=5, Mag_elaps=[5, 6], lstm_feature_mode="external_sliding")
    c = _load_lstm_event_windows_with_cache(args, catalog_dir)
    args.time_step = 10
    args.Mag_elaps = [6, 7]
    _load_lstm_event_windows_with_cache(args, catalog_dir)
    assert len(calls) == 1
    assert len(list((catalog_dir / "cache/windows").glob("*.pkl"))) == 1
    assert a.df.attrs["window_cache_path"] == b.df.attrs["window_cache_path"] == c.df.attrs["window_cache_path"]
    assert [s["t"] for s in a.samples_list] == [s["t"] for s in c.samples_list]


def test_numeric_spelling_reuses_cache_but_fit_boundary_does_not(catalog_dir):
    a = event_pipeline.load_event_windows_with_cache(catalog_dir, **options())
    b = event_pipeline.load_event_windows_with_cache(catalog_dir, **{k: float(v) if k != "context_len" else v for k, v in options().items()})
    c = event_pipeline.load_event_windows_with_cache(catalog_dir, **options(), fit_end=20)
    assert a.df.attrs["window_cache_path"] == b.df.attrs["window_cache_path"]
    assert a.df.attrs["window_cache_path"] != c.df.attrs["window_cache_path"]


def test_failed_cache_write_does_not_publish_partial_file(tmp_path, monkeypatch):
    def fail(data, stream):
        stream.write(b"partial")
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(file_utils.pickle, "dump", fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            file_utils.save_or_load_data(tmp_path, lambda: {"value": 1}, filename="sample.pkl")
    assert not (tmp_path / "processed/sample.pkl").exists()
    assert not list((tmp_path / "processed").glob(".cache-*"))
    assert file_utils.save_or_load_data(tmp_path, lambda: {"value": 2}, filename="sample.pkl") == {"value": 2}
