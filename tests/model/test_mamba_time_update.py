"""Physical-gap modulation must preserve the intended ADT/DT distinction."""
import copy
import pytest
import torch
from src.models.mamba.time_update import apply_time_update


@pytest.mark.parametrize('mode', ['base', 'full_dt', 'decay_only'])
def test_unit_gap_recovers_original_coefficients(mode):
    torch.manual_seed(101)
    A = -torch.rand(2, 7, 4, dtype=torch.float64)
    dt = torch.rand(2, 7, 4, dtype=torch.float64)
    adt, effective_dt = apply_time_update(A, dt, torch.ones(2, 7), mode)
    torch.testing.assert_close(adt, A*dt, rtol=0, atol=0)
    torch.testing.assert_close(effective_dt, dt, rtol=0, atol=0)


def test_decay_only_changes_retention_without_scaling_write_or_native_angle_step():
    A = torch.full((1, 3, 2), -.7, dtype=torch.float64, requires_grad=True)
    dt = torch.full((1, 3, 2), .2, dtype=torch.float64, requires_grad=True)
    gaps = torch.tensor([[.1, 1., 10.]], dtype=torch.float64, requires_grad=True)
    adt, effective_dt = apply_time_update(A, dt, gaps, 'decay_only')
    assert effective_dt is dt
    retention = adt.exp()
    assert torch.all(retention[:, 0] > retention[:, 1])
    assert torch.all(retention[:, 1] > retention[:, 2])
    retention.sum().backward()
    assert torch.isfinite(A.grad).all() and torch.isfinite(dt.grad).all()
    assert torch.isfinite(gaps.grad).all() and torch.all(gaps.grad < 0)


def test_full_dt_scales_native_rotation_and_current_write_step():
    A = torch.full((1, 3, 2), -.7)
    dt = torch.full((1, 3, 2), .2)
    gaps = torch.tensor([[.1, 1., 10.]])
    adt, effective_dt = apply_time_update(A, dt, gaps, 'full_dt')
    torch.testing.assert_close(effective_dt / dt, gaps[..., None].expand_as(dt))
    torch.testing.assert_close(adt, A*effective_dt)
    assert not torch.equal(effective_dt, dt)
    with pytest.raises(ValueError, match='required'):
        apply_time_update(A, dt, None, 'full_dt')


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='Native GPU kernels required')


@GPU
@pytest.mark.parametrize('mimo', [False, True])
@pytest.mark.parametrize('mode', ['full_dt', 'decay_only'])
def test_native_scan_gradient_and_cached_step_support_physical_gaps(mimo, mode):
    from src.models.mamba.mamba3_rotary import Mamba3Rotary
    from mamba_ssm.utils.generation import InferenceParams
    torch.manual_seed(103)
    model = Mamba3Rotary(d_model=64, d_state=64, expand=2, headdim=32,
        is_mimo=mimo, mimo_rank=2, chunk_size=16 if mimo else 64, layer_idx=0,
        time_update_mode=mode, device='cuda', dtype=torch.bfloat16)
    reference = copy.deepcopy(model)
    reference.time_update_mode = 'base'
    features = torch.randn(2, 65, 64, device='cuda', dtype=torch.bfloat16)
    gaps = torch.rand(2, 65, device='cuda', dtype=torch.float64) * 1.5 + .05
    times = gaps.cumsum(1)
    with torch.no_grad():
        original = reference(features, times)
        unit_gap = model(features, times, physical_dt=torch.ones_like(gaps))
        torch.testing.assert_close(original, unit_gap, rtol=0, atol=0)
    output = model(features, times, physical_dt=gaps)
    (output.float() * torch.randn_like(output).float()).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.in_proj.weight.grad[-model.num_rope_angles:].abs().sum() > 0
    cache = InferenceParams(max_seqlen=100, max_batch_size=2)
    model.eval()
    with torch.no_grad():
        expected = model(features, times, physical_dt=gaps)
        pieces = [model(features[:, :59], times[:, :59], physical_dt=gaps[:, :59], inference_params=cache)]
        cache.seqlen_offset = 59
        pieces.append(model(features[:, 59:63], times[:, 59:63], physical_dt=gaps[:, 59:63], inference_params=cache))
        for i in [63, 64]:
            cache.seqlen_offset = i
            pieces.append(model(features[:, i:i+1], times[:, i:i+1], physical_dt=gaps[:, i:i+1], inference_params=cache))
        actual = torch.cat(pieces, 1)
    error = (actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-8)
    assert error < .03
