from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.cli.common import resolve_test_threshold
from src.data import event_loader, lstm_loader
from src.data.event_pipeline import load_event_windows_with_cache
from src.data.preparation_modules.common import _target_period_indices
from src.data.preparation_modules.lstm import _normalize_lstm_inputs


def args():
    return SimpleNamespace(Tfore=3, split_by_time=True, split_dates={
        'train_end': '1970-01-11', 'validation_end': '1970-01-21',
        'test_end': '1970-01-31'})


def catalog():
    return pd.DataFrame({'t': [9., 11.], 'Magnitude': [5., 3.],
                         'Latitude': [0., 1.], 'Longitude': [0., 1.],
                         'Depth': [1., 1.], 'dt': [0., 2.]})


def test_existing_context_zero_keeps_a_future_negative_after_a_large_event():
    df = catalog()
    zero = event_loader.generate_single_sample(df, df, 10., 5., 2., 4., 0)
    strict = event_loader.generate_single_sample(df, df, 10., 5., 2., 4., 1)
    label = lambda x: event_loader.compute_classification_label(
        np.array([e['t'] for e in x['future_dict']]),
        np.array([e['Magnitude'] for e in x['context_dict']]), 4.)
    assert label(zero) == 0
    assert np.isnan(label(strict))
    df.loc[1, 'Magnitude'] = 5.
    positive = event_loader.generate_single_sample(df, df, 10., 5., 2., 4., 0)
    assert label(positive) == 1


def test_target_periods_have_disjoint_future_events_and_allow_past_history():
    origins = np.arange(1., 29.)
    groups = _target_period_indices(origins, args())
    events = np.arange(0., 31., .25)
    event_sets = []
    for idx in groups:
        mask = np.any((events[None, :] >= origins[idx, None]) &
                      (events[None, :] < origins[idx, None] + 3), axis=0)
        event_sets.append(set(np.flatnonzero(mask)))
    assert not event_sets[0] & event_sets[1]
    assert not event_sets[1] & event_sets[2]
    assert origins[groups[1][0]] == 10
    # A validation history can reach into training time; only future targets purge.
    assert origins[groups[1][0]] - 8 < 10
    assert not any(8 in origins[i] or 9 in origins[i] for i in groups)


def test_catalog_scaler_cannot_learn_future_extrema():
    df = catalog()
    df.loc[2] = [25., 3., 1000., 1000., 1000., 14.]
    transformed, scalers = event_loader.normalize_df(df, fit_df=df.iloc[:2])
    assert scalers['Latitude'].data_max_[0] == 1
    assert transformed.Latitude.iloc[-1] == 1000


def test_lstm_training_scaling_is_invariant_to_holdout_changes():
    x = np.array([[[0.], [1.]], [[2.], [3.]], [[100.], [200.]]])
    first, a = _normalize_lstm_inputs(x, ['feature'], lstm_loader, fit_indices=[0, 1])
    x[-1] *= 10000
    second, b = _normalize_lstm_inputs(x, ['feature'], lstm_loader, fit_indices=[0, 1])
    np.testing.assert_array_equal(first[:2], second[:2])
    np.testing.assert_array_equal(a['feature'].data_max_, b['feature'].data_max_)
    assert a['feature'].data_max_[0] == 3


def test_new_protocol_refuses_test_threshold_fallback():
    cli = SimpleNamespace(threshold=None, no_val_threshold=False)
    checkpoint = {'hyperparameters': {'split_dates': args().split_dates}}
    with pytest.raises(ValueError, match='validation threshold'):
        resolve_test_threshold(cli, checkpoint)
    checkpoint['val_metrics'] = {'threshold': .37}
    assert resolve_test_threshold(cli, checkpoint) == .37
    cli.no_val_threshold = True
    with pytest.raises(ValueError):
        resolve_test_threshold(cli, checkpoint)
    assert resolve_test_threshold(SimpleNamespace(threshold=None, no_val_threshold=False), {}) is None


def test_versioned_cache_does_not_reuse_changed_raw_data(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    df = pd.concat([catalog()] * 4, ignore_index=True)
    df['t'] = np.arange(1., 17., 2.)
    path = raw / 'events.csv'
    df.to_csv(path, index=False)
    kw = dict(Mc=3, Mf=4, Twindow=2, Tfore=2, dt=2,
              context_len=0, task_prefix='classifier', fit_end=10,
              cache_namespace='review_v2')
    load_event_windows_with_cache(tmp_path, **kw)
    previous = set((tmp_path / 'classifier').glob('*.pkl'))
    df.loc[len(df) - 1, 'Magnitude'] = 7.
    df.to_csv(path, index=False)
    load_event_windows_with_cache(tmp_path, **kw)
    current = set((tmp_path / 'classifier').glob('*.pkl'))
    assert len(previous) == 1 and len(current) == 2
    assert previous < current
