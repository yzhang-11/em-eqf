"""Phase interventions must not change the real inputs or recurrence dynamics."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/run'))
from run_mamba_phase_ablation import PhaseClockAdapter, surrogate_clock
from src.data.dot_dict import DotDict
from src.models.mamba.mamba3_rotary import Mamba3Rotary
from mamba_ssm.utils.generation import InferenceParams


@pytest.mark.parametrize('mode', ['native', 'index', 'real', 'surrogate'])
def test_clock_adapter_preserves_every_real_data_output(mode):
    real = dict(features=torch.randn(2, 7, 2), times=torch.randn(2, 7, dtype=torch.float64),
                input_mask=torch.ones(2, 7))
    batch = DotDict(phase_clock=torch.arange(7, dtype=torch.float64)[None].expand(2, -1))
    adapter = PhaseClockAdapter(lambda _: dict(real), mode)
    result = adapter(batch)
    assert all(result[key] is value for key, value in real.items())
    if mode == 'native':
        assert torch.count_nonzero(result['phase_times']) == 0
    if mode == 'real':
        assert 'phase_times' not in result


def test_surrogate_clock_does_not_read_validation_or_future_intervals():
    data = dict(regions={'r':dict(metadata=[0., 1., 2.], splits={
        'train':dict(times=np.array([0., .2, .7, 1., 2.])),
        'val':dict(times=np.array([0., .1, .9, 3., 4.])),
        'test':dict(times=np.array([0., 1., 2., 4., 9.]))})})
    before = surrogate_clock(data, 'r', 'val', 0)
    changed = copy.deepcopy(data)
    changed['regions']['r']['splits']['val']['times'][2:] *= 1000.
    changed['regions']['r']['splits']['test']['times'] *= 30.
    after = surrogate_clock(changed, 'r', 'val', 0)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(before[:3], surrogate_clock(data, 'r', 'val', 0, length=3))
    assert np.all(np.diff(before) > 0)
    assert not np.array_equal(before, surrogate_clock(data, 'r', 'val', 1))


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='Native Mamba3 GPU kernels')


def layer(mimo):
    return Mamba3Rotary(d_model=64, d_state=64, expand=2, headdim=32,
                       is_mimo=mimo, mimo_rank=2, chunk_size=16 if mimo else 64,
                       layer_idx=0, device='cuda', dtype=torch.bfloat16)


@GPU
@pytest.mark.parametrize('mimo', [False, True])
def test_override_is_an_independent_differentiable_phase_path(mimo):
    torch.manual_seed(71)
    model = layer(mimo)
    features = torch.randn(2, 65, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    actual = torch.arange(65, device='cuda', dtype=torch.float64)[None].expand(2, -1).clone().requires_grad_()
    override = (actual.detach() * .23).clone().requires_grad_()
    projected = model._project(features)
    output = model(features, actual, phase_times=override)
    explicit = model(features, override)
    torch.testing.assert_close(output, explicit, rtol=0, atol=0)
    assert all(torch.equal(a, b) for a, b in zip(projected, model._project(features)))
    weights = torch.randn_like(output)
    (output.float() * weights.float()).mean().backward()
    assert actual.grad is None
    assert override.grad is not None and torch.isfinite(override.grad).all()
    assert override.grad.abs().sum() > 0
    assert features.grad is not None and torch.isfinite(features.grad).all()
    # The learned native-angle projection remains active under the override.
    assert model.in_proj.weight.grad[-model.num_rope_angles:].abs().sum() > 0


@GPU
@pytest.mark.parametrize('mimo', [False, True])
@pytest.mark.parametrize('clock_mode', ['zero', 'index', 'surrogate'])
def test_override_matches_full_scan_during_cached_continuation(mimo, clock_mode):
    torch.manual_seed(73)
    model = layer(mimo).eval()
    features = torch.randn(2, 65, 64, device='cuda', dtype=torch.bfloat16)
    actual = torch.linspace(.1, 50., 65, device='cuda', dtype=torch.float64)[None].expand(2, -1)
    if clock_mode == 'zero':
        override = torch.zeros_like(actual)
    elif clock_mode == 'index':
        override = torch.arange(65, device='cuda', dtype=torch.float64)[None].expand(2, -1)
    else:
        override = (torch.rand(2, 65, device='cuda', dtype=torch.float64) + .01).cumsum(1)
    cache = InferenceParams(max_seqlen=100, max_batch_size=2)
    with torch.no_grad():
        full = model(features, actual, phase_times=override)
        outputs = [model(features[:, :59], actual[:, :59], phase_times=override[:, :59], inference_params=cache)]
        cache.seqlen_offset = 59
        outputs.append(model(features[:, 59:63], actual[:, 59:63], phase_times=override[:, 59:63], inference_params=cache))
        for i in [63, 64]:
            cache.seqlen_offset = i
            outputs.append(model(features[:, i:i+1], actual[:, i:i+1], phase_times=override[:, i:i+1], inference_params=cache))
        cached = torch.cat(outputs, 1)
    error = (cached.float() - full.float()).norm() / full.float().norm().clamp_min(1e-8)
    assert error < .03
