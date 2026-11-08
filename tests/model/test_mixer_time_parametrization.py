"""Mixer time heads retain legacy likelihoods and support log-mean decoding."""
import math

import pytest
import torch
from omegaconf import OmegaConf

from src.data import Batch, Sequence
from src.models.builders import ModelBuilder


def _model(mode=None, *, n_layer=1):
    args = OmegaConf.load("config/mixer_tpp.yaml")
    args.pop("mixer_time_parametrization", None)
    if mode is not None:
        args.mixer_time_parametrization = mode
    args.d_model = 8
    args.num_components = 2
    args.dropout = 0.0
    args.predict_b = False
    args.bg_model = None
    args.mixer_model_config = dict(
        input_dim=2, d_model=8, n_layer=n_layer, d_intermediate=0,
        attn_layer_idx=[0], attn_cfg=dict(num_heads=2, causal=True),
        fused_add_norm=False, rms_norm=False, dropout_prob=0.0,
    )
    for key, value in dict(
        tau_mean=1.0, tau_min=0.0, tau_max=3.0, tau_q05=0.1,
        tau_q025=0.05, mag_mean=2.4, time_max=10.0,
        richter_b_mle=1.0, mag_completeness=2.0,
    ).items():
        args[key] = value
    return ModelBuilder.by_name("mixer_tpp")()(args, torch.device("cpu")).eval()


def _sequence():
    return Sequence(
        inter_times=torch.tensor([0.3, 0.7, 0.4]),
        mag=torch.tensor([2.2, 2.4]),
    )


def _batch():
    return Batch.from_list([_sequence()])


def test_default_preserves_legacy_likelihood_and_checkpoint_shapes():
    torch.manual_seed(12)
    default = _model()
    explicit = _model("legacy")
    explicit.load_state_dict(default.state_dict(), strict=True)
    assert default.inter_time_decoder.parametrization == "legacy"
    batch = _batch()
    for key, value in default.nll_loss(batch).items():
        torch.testing.assert_close(value, explicit.nll_loss(batch)[key], rtol=0, atol=0)

    context = default.get_context(batch)
    raw_rate, raw_shape, logits = default.hypernet_time(context).chunk(3, dim=-1)
    rate = torch.nn.functional.softplus(raw_rate.clamp_min(-5.0))
    shape = torch.nn.functional.softplus(raw_shape.clamp_min(-5.0))
    reference = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.Weibull(scale=rate.pow(-shape.reciprocal()), concentration=shape),
    )
    torch.testing.assert_close(
        default.get_inter_time_dist(context).log_prob(batch.inter_times),
        reference.log_prob(batch.inter_times),
    )


def test_oracle_component_means_match_requested_means():
    model = _model("oracle")
    means = torch.tensor([2.0, 0.5])
    raw_means = torch.atanh(means.log() / (5.0 * math.log(10.0)))
    with torch.no_grad():
        model.hypernet_time.weight.zero_()
        model.hypernet_time.bias.copy_(torch.cat([
            raw_means, torch.full((2,), math.log(math.expm1(1.0))), torch.zeros(2),
        ]))
    dist = model.get_inter_time_dist(torch.zeros(1, 3, 8))
    torch.testing.assert_close(dist.component_distribution.mean, means.expand(1, 3, 2))


@pytest.mark.parametrize("mode", ["legacy", "oracle"])
def test_both_modes_have_finite_loss_and_gradients(mode):
    torch.manual_seed(24)
    model = _model(mode)
    with torch.no_grad():
        model.hypernet_time.weight.zero_()
        model.hypernet_time.bias.zero_()
        model.hypernet_time.bias[2:4].fill_(math.log(math.expm1(1.0)))
    out = model.nll_loss(_batch())
    assert all(torch.isfinite(value).all() for value in out.values())
    out["total"].mean().backward()
    grad = model.hypernet_time.bias.grad
    assert torch.isfinite(grad).all()
    assert grad[:2].abs().sum() > 0
    assert grad[2:4].abs().sum() > 0


@pytest.mark.parametrize("mode", ["legacy", "oracle"])
def test_both_modes_support_historical_sampling(mode):
    # The full sampler can run on CPU with a projection-only mixer backbone;
    # cached attention and Mamba kernels themselves require CUDA.
    torch.manual_seed(24)
    model = _model(mode, n_layer=0)
    with torch.no_grad():
        model.hypernet_time.weight.zero_()
        model.hypernet_time.bias.zero_()
        model.hypernet_time.bias[2:4].fill_(math.log(math.expm1(1.0)))
    history = _sequence()
    generated = model.sample(batch_size=2, duration=0.25, past_seq=history, max_sample_len=32)
    assert generated.batch_size == 2
    assert torch.isfinite(generated.inter_times).all()
    assert (generated.inter_times >= 0).all()
    torch.testing.assert_close(generated.t_end, torch.full_like(generated.t_end, float(history.t_end) + 0.25))


def test_invalid_parametrization_fails_at_model_creation():
    with pytest.raises(ValueError, match="parametrization"):
        _model("unknown")
