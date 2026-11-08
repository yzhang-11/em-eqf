"""Mamba3 GPU environment integration for the supported small MIMO configuration."""
import pytest
import torch
from mamba_ssm.utils.generation import InferenceParams
from src.models.mamba.mamba3_rotary import Mamba3Rotary
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def layer(**kwargs):
    options=dict(d_model=64,d_state=64,expand=2,layer_idx=0)
    options.update(kwargs)
    return Mamba3Rotary(**options)
def event_times(length):
    gaps=torch.where(torch.arange(length,device="cuda")%7==0,1.7,.03).double()
    return gaps.cumsum(0)[None].expand(2,-1).contiguous()
def relative_error(actual,expected):
    return float(((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-8)).detach())


@CUDA
def test_mimo_event_time_scan_backward_and_cached_sampling():
    torch.manual_seed(51)
    model = layer(device="cuda", dtype=torch.bfloat16, is_mimo=True,
                  mimo_rank=2, headdim=32, chunk_size=16)
    x = torch.randn(2, 65, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    times = event_times(65).clone().requires_grad_(True)
    output = model(x, times)
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(times.grad).all()
    assert times.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    # All simulated paths must inherit the prefix, including MIMO K/V states.
    model.eval()
    prefix = x[:1].detach()
    clock = times[:1].detach()
    cache = InferenceParams(max_seqlen=100, max_batch_size=3,
        key_value_memory_dict={0: model.allocate_inference_cache(3, 100)})
    with torch.no_grad():
        expected = model(prefix, clock)[:, -1:].expand(3, -1, -1)
        model(prefix[:, :64], clock[:, :64], inference_params=cache)
        cache.seqlen_offset = 64
        actual = model(prefix[:, 64:].expand(3, -1, -1), clock[:, 64:].expand(3, -1), inference_params=cache)
    assert relative_error(actual, expected) < .03
    for state in cache.key_value_memory_dict[0]:
        torch.testing.assert_close(state[1], state[0])
        torch.testing.assert_close(state[2], state[0])
