from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest
import torch

from src.data import Batch, Sequence
from src.data.preprocessing import resolve_tied_times
from src.catalogs.chuandian import ChuanDianStandard
from src.catalogs.azdx import AZDXStandard
from src.catalogs.scedc import SCEDC
from src.models.adapters.mixer_tpp import MixerBatchAdapter
from src.models.mamba.mixer_seq import MixerModelWrapper
from src.models.mha.rotary_embedding_time import RotaryEmbeddingTime
from src.models.tpp.mixer_tpp import MixerTPP


def test_tie_resolution_preserves_distinct_times_order_and_recording_bin():
    original = np.array([20000., 20000., 20000., 20000.0001, 20001.])
    actual = resolve_tied_times(original, .01, 20002.)
    np.testing.assert_array_equal(actual[[0, 3, 4]], original[[0, 3, 4]])
    assert np.all(np.diff(actual) > 0)
    assert actual[2] < original[3]
    np.testing.assert_array_equal(actual, resolve_tied_times(original, .01, 20002.))
    with pytest.raises(ValueError):
        resolve_tied_times([2., 1.], .01, 3.)


def test_swc_keeps_first_and_tied_events(tmp_path):
    source = tmp_path / 'swc.dat'
    source.write_text('1970 1 1 0 0 1.00 30 100 10 3.0\n'
                      '1970 1 1 0 0 1.00 31 101 20 4.0\n'
                      '1970 1 1 0 0 1.01 32 102 30 5.0\n')
    catalog = ChuanDianStandard(root_dir=tmp_path / 'cache', catalog_file=source, mag_completeness=3.)
    seq = catalog.full_sequence
    assert seq.num_events == 3
    assert seq.inter_times.dtype == torch.float64
    assert torch.all(torch.diff(seq.arrival_times) > 0)
    np.testing.assert_allclose(seq.mag, [3., 4., 5.])


def test_azdx_ties_are_deterministic_and_count_preserving(tmp_path):
    source = tmp_path / 'azdx.dat'
    source.write_text('1 0 116.000 5.1 100 101 30\n'
                      '2 0 116.000 5.2 200 102 31\n'
                      '3 0 116.001 5.3 300 103 32\n')
    a = AZDXStandard(root_dir=tmp_path / 'cache_a', catalog_file=source, mag_completeness=4.75)
    b = AZDXStandard(root_dir=tmp_path / 'cache_b', catalog_file=source, mag_completeness=4.75)
    torch.testing.assert_close(a.full_sequence.inter_times, b.full_sequence.inter_times, rtol=0, atol=0)
    assert a.full_sequence.num_events == 3
    assert torch.all(a.full_sequence.inter_times[:-1] > 0)


def test_scsn_preserves_both_records_at_same_time(tmp_path):
    source = tmp_path / 'events.csv'
    pd.DataFrame({'ts': ['1981-01-02', '1981-01-02', '1981-01-03'],
                  'magnitude': [2.5, 2.6, 3.]}).to_csv(source, index=False)
    catalog = SCEDC(root_dir=tmp_path / 'cache', catalog_file=source, mag_completeness=2.5)
    assert catalog.full_sequence.num_events == 3
    assert torch.all(torch.diff(catalog.full_sequence.arrival_times) > 0)


def test_exact_split_boundary_events_are_targets_of_only_one_period(tmp_path):
    source = tmp_path / 'events.csv'
    pd.DataFrame({'ts': ['1981-01-02', '1981-01-03', '1981-01-04'],
                  'magnitude': [3., 4., 5.]}).to_csv(source, index=False)
    catalog = SCEDC(root_dir=tmp_path / 'cache', catalog_file=source, mag_completeness=2.5,
                    train_start_ts='1981-01-01', val_start_ts='1981-01-03', test_start_ts='1981-01-04')
    targets = [seq.mag[seq.arrival_times >= seq.t_nll_start].tolist()
               for seq in [catalog.train[0], catalog.val[0], catalog.test[0]]]
    assert targets == [[3.], [4.], [5.]]


def test_mixer_keeps_clock_precision_without_mutating_batch():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, 2)

        def forward(self, features, times, **kwargs):
            assert features.dtype == torch.float32
            assert times.dtype == torch.float64
            assert times[0, 1] > times[0, 0]
            return self.projection(features)

    args = OmegaConf.create(dict(tau_mean=1., tau_min=1e-8, tau_max=10., tau_q05=1., tau_q025=.1,
                                 mag_mean=4., time_max=30000., richter_b_mle=1., mag_completeness=3.,
                                 features_input_keys=['mag'], normalize_time_by_token=True))
    batch = Batch.from_list([Sequence(torch.tensor([20000., 1e-8, 1.], dtype=torch.float64),
                                      mag=torch.tensor([3., 4.]))])
    before = batch.arrival_times.clone()
    wrapper = MixerModelWrapper(Encoder(), MixerBatchAdapter(args), torch.device('cpu'))
    assert wrapper(batch).dtype == torch.float32
    torch.testing.assert_close(batch.arrival_times, before, rtol=0, atol=0)
    assert batch.inter_times.dtype == torch.float64


def test_rotary_uses_double_clock_and_accepts_integer_positions():
    rope = RotaryEmbeddingTime(4)
    qkv = torch.ones(1, 2, 3, 1, 4)
    times = torch.tensor([[20000., 20000.0001]], dtype=torch.float64)
    actual = rope(qkv, times=times)
    assert actual.dtype == torch.float32
    assert (actual[:, 0] - actual[:, 1]).abs().max() > 1e-6
    assert torch.isfinite(rope(qkv, times=torch.arange(2))).all()


def test_conditional_sampling_horizon_shorter_than_elapsed_history():
    class Distribution:
        def sample_conditional(self, lower_bound):
            return lower_bound.reshape(1, 1) + .5

        def sample(self):
            return torch.tensor([[3.]])

    class Model:
        predict_b = True
        b_filter = None
        device = torch.device('cpu')
        base_model = SimpleNamespace(encoder=SimpleNamespace(allocate_inference_cache=lambda **kwargs: {}))

        def get_current_state(self, batch, **kwargs):
            return torch.zeros(batch.inter_times.shape[0], batch.inter_times.shape[1], 1)

        def get_inter_time_dist(self, context):
            return Distribution()

        def get_magnitude_dist(self, **kwargs):
            return Distribution()

    past = Sequence(torch.tensor([20000., 10.], dtype=torch.float64), mag=torch.tensor([3.], dtype=torch.float64))
    sample = MixerTPP.sample(Model(), batch_size=1, duration=.1, past_seq=past, max_sample_len=2)
    assert sample.end_idx.item() == 0
    assert sample.arrival_times.dtype == torch.float64
    assert (sample.inter_times >= 0).all()
