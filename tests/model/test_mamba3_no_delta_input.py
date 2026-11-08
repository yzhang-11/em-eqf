"""Removing the explicit interval must leave the phase clock and labels intact."""
from pathlib import Path
import sys
import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/run'))
from run_mamba3_mimo_no_deltat import NoDeltaTAdapter


@pytest.mark.parametrize('feature_order', [['mag', 'log_inter_times'], ['log_inter_times', 'mag']])
def test_interval_information_removed_with_real_clock_and_marks_preserved(feature_order):
    features = torch.randn(2, 9, 2, dtype=torch.float64)
    clock = torch.rand(2, 9, dtype=torch.float64).cumsum(1)
    labels = torch.rand(2, 9, dtype=torch.float64)

    class Delegate:
        features_input_keys = feature_order
        def __call__(self, batch):
            return dict(features=batch['features'], times=clock, input_mask=torch.ones(2, 9))

    adapter = NoDeltaTAdapter(Delegate())
    original = features.clone()
    dt_index, mag_index = feature_order.index('log_inter_times'), feature_order.index('mag')
    result = adapter(dict(features=features, labels=labels))
    changed = features.clone()
    changed[..., dt_index] = 1000 * torch.randn_like(changed[..., dt_index])
    other = adapter(dict(features=changed, labels=labels))
    torch.testing.assert_close(result['features'], other['features'], rtol=0, atol=0)
    torch.testing.assert_close(result['features'][..., mag_index], features[..., mag_index], rtol=0, atol=0)
    assert torch.count_nonzero(result['features'][..., dt_index]) == 0
    assert result['times'] is clock and other['times'] is clock
    torch.testing.assert_close(features, original, rtol=0, atol=0)
