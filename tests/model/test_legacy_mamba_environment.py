"""Numerical and recurrent checks of the optional legacy CUDA interface."""
import copy

import pytest
import torch
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.utils.generation import InferenceParams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_legacy_scan_output_and_gradients_match_reference():
    torch.manual_seed(61)
    batch, channels, state, length = 2, 32, 16, 129
    tensors = [torch.randn(batch, channels, length, device='cuda'),
               torch.randn(batch, channels, length, device='cuda')*.1,
               -torch.rand(channels, state, device='cuda'),
               torch.randn(batch, state, length, device='cuda'),
               torch.randn(batch, state, length, device='cuda'),
               torch.randn(channels, device='cuda')]
    actual = [t.clone().requires_grad_(True) for t in tensors]
    reference = [t.clone().requires_grad_(True) for t in tensors]
    y, final = selective_scan_fn(*actual, delta_softplus=True, return_last_state=True)
    expected, expected_final = selective_scan_ref(*reference, delta_softplus=True, return_last_state=True)
    torch.testing.assert_close(y, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(final, expected_final, rtol=2e-4, atol=2e-4)
    weight = torch.randn_like(y)
    (y*weight).mean().backward()
    (expected*weight).mean().backward()
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-3, atol=2e-5)


def test_legacy_mamba_training_and_cached_decode():
    torch.manual_seed(62)
    model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2, layer_idx=0, device='cuda')
    x = torch.randn(2, 65, 64, device='cuda', requires_grad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    y = model(x)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    model.eval()
    cache = InferenceParams(max_seqlen=100, max_batch_size=2)
    with torch.no_grad():
        expected = model(x)
        prefix = model(x[:, :64], inference_params=cache)
        cache.seqlen_offset = 64
        last = model(x[:, 64:], inference_params=cache)
    torch.testing.assert_close(torch.cat([prefix, last], 1), expected, rtol=1e-3, atol=2e-4)
