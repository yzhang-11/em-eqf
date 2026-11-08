"""Residual toggles preserve defaults and share the full/cached forward path."""
from functools import partial

import pytest
import torch
from torch import nn
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.utils.generation import InferenceParams
from omegaconf import OmegaConf

from src.models.builders import ModelBuilder
from src.models.mamba.block import Block
from src.models.mamba.mamba2_aligned import AlignedMamba2
from src.models.mamba.mixer_seq import MixerModel


class ZeroMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x, inference_params=None):
        return torch.zeros_like(x)


@pytest.mark.parametrize("with_mlp", [False, True])
@pytest.mark.parametrize("use_residual", [False, True])
def test_residual_toggle_controls_bypass_when_main_branch_is_zero(with_mlp, use_residual):
    mlp = partial(nn.Linear, out_features=8, bias=False) if with_mlp else nn.Identity
    model = Block(8, ZeroMixer, mlp, use_residual=use_residual)
    if with_mlp:
        with torch.no_grad():
            model.mlp.weight.zero_()
    x = torch.randn(2, 5, 8, requires_grad=True)
    main, residual = model(x)
    assert torch.count_nonzero(main) == 0
    if use_residual:
        torch.testing.assert_close(residual, x, rtol=0, atol=0)
        residual.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
    else:
        assert residual is None


def _cpu_mixer(**options):
    return MixerModel(d_model=8, n_layer=2, d_intermediate=16, input_dim=2,
                      attn_layer_idx=[0, 1], attn_cfg=dict(num_heads=2, causal=True),
                      fused_add_norm=False, **options).eval()


@pytest.mark.parametrize("configured,expected", [(None, False), (False, False), (True, True)])
def test_mixer_tpp_defaults_to_disabled_and_honors_explicit_flag(configured, expected):
    settings = dict(input_dim=2, d_model=8, n_layer=0, d_intermediate=0)
    if configured is not None:
        settings["use_residual"] = configured
    args = OmegaConf.create(dict(
        d_model=8, num_components=2, dropout=0.0, predict_b=False,
        mixer_model_config=settings, tau_mean=1.0, tau_min=0.0, tau_max=3.0,
        tau_q05=0.1, tau_q025=0.05, mag_mean=2.4, time_max=10.0,
        richter_b_mle=1.0, mag_completeness=2.0,
    ))
    model = ModelBuilder.by_name("mixer_tpp")()(args, torch.device("cpu"))
    assert model.base_model.encoder.use_residual is expected
    if configured is None:
        assert "use_residual" not in args.mixer_model_config


def test_default_and_explicit_enabled_have_identical_outputs_and_checkpoint_keys():
    default = _cpu_mixer()
    enabled = _cpu_mixer(use_residual=True)
    disabled = _cpu_mixer(use_residual=False)
    enabled.load_state_dict(default.state_dict(), strict=True)
    disabled.load_state_dict(default.state_dict(), strict=True)
    x = torch.randn(2, 7, 2)
    torch.testing.assert_close(default(x), enabled(x), rtol=0, atol=0)
    assert not torch.allclose(default(x), disabled(x))
    assert all(not block.use_residual for block in disabled.layers)


@pytest.mark.parametrize("use_residual", [False, True])
def test_future_inputs_do_not_change_past_outputs(use_residual):
    model = _cpu_mixer(use_residual=use_residual)
    x = torch.randn(2, 7, 2)
    changed = x.clone()
    changed[:, 4:] += 5.0
    torch.testing.assert_close(model(x)[:, :4], model(changed)[:, :4], rtol=0, atol=0)


def test_mamba2_projection_alignment_preserves_weights_values_and_gradients():
    reference = Mamba2(d_model=64, headdim=64, d_state=128)
    aligned = AlignedMamba2(d_model=64, headdim=64, d_state=128)
    aligned.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 11, 64, requires_grad=True)
    actual = aligned.in_proj(x)
    expected = reference.in_proj(x.detach())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride(-2) % 8 == 0
    assert actual.shape == expected.shape
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Mamba CUDA kernels")
@pytest.mark.parametrize("use_residual", [False, True])
@pytest.mark.parametrize("fused_add_norm", [False, True])
def test_mamba2_residual_full_and_cached_inference_agree(use_residual, fused_add_norm):
    torch.manual_seed(31)
    model = MixerModel(d_model=64, n_layer=2, d_intermediate=128, input_dim=2,
                       ssm_cfg=dict(layer="Mamba2", d_state=128, headdim=64),
                       use_residual=use_residual, residual_in_fp32=True,
                       fused_add_norm=fused_add_norm, device="cuda").eval()
    x = torch.randn(2, 35, 2, device="cuda")
    cache = InferenceParams(max_seqlen=64, max_batch_size=2,
                           key_value_memory_dict=model.allocate_inference_cache(2, 64))
    with torch.inference_mode():
        full = model(x)
        pieces = [model(x[:, :31], inference_params=cache)]
        for i in range(31, 35):
            cache.seqlen_offset = i
            pieces.append(model(x[:, i:i+1], inference_params=cache))
        cached = torch.cat(pieces, dim=1)
    assert torch.isfinite(cached).all()
    # Fused full scans and single-step kernels have different floating-point
    # accumulation orders; check overall and worst absolute error explicitly.
    diff = cached.float() - full.float()
    relative_error = diff.norm() / full.float().norm().clamp_min(1e-8)
    assert relative_error < 1e-3
    assert diff.abs().max() < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Mamba CUDA kernels")
@pytest.mark.parametrize("use_residual", [False, True])
def test_aligned_mamba2_full_scan_keeps_finite_training_gradients(use_residual):
    torch.manual_seed(43)
    model = MixerModel(d_model=64, n_layer=1, d_intermediate=128, input_dim=2,
                       ssm_cfg=dict(layer="Mamba2", d_state=128, headdim=64),
                       use_residual=use_residual, device="cuda")
    x = torch.randn(2, 35, 2, device="cuda", requires_grad=True)
    output = model(x)
    (output * torch.randn_like(output)).mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
    gradient = model.layers[0].mixer.in_proj.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
