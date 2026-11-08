import math
import copy

import pytest
import torch
from torch import nn

from src.data import Batch, Sequence
from src.data.dot_dict import DotDict
from src.models.adapters.mixer_tpp import MixerBatchAdapter
from src.models.tpp.magnitude_density import MagnitudeGridDistribution, inverse_positive
from src.models.tpp.mixer_tpp import MixerTPP
from src.models.tpp.region_mixer_tpp import (
    RegionalBatchAdapter, RegionalGridGR, RegionalMixerTPP, attach_region,
)


def args():
    return DotDict(tau_mean=1., tau_min=0., tau_max=3., tau_q05=.1,
                   tau_q025=.05, mag_mean=2.4, time_max=100.,
                   richter_b_mle=1., mag_completeness=2.,
                   features_input_keys=['mag', 'log_inter_times'],
                   normalize_time_by_token=True, time_scale_base_key='tau_mean')


def sequence(mc=2., tau=1., n=4, step=.1, kind=1):
    inter = torch.tensor([.3, .7, .4, .6, .2][:n+1], dtype=torch.float64) * tau
    seq = Sequence(inter, mag=mc + step * torch.arange(1, n+1, dtype=torch.float64))
    return attach_region(seq, [mc, 1., tau, 0., 3*tau, step, kind])


class TinyBase(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.encoder = nn.Linear(2, 8)
        self.encoder.d_model = 8
        self.input_adapter = MixerBatchAdapter(settings)
        self.device = torch.device('cpu')

    def forward(self, batch, inference_params=None):
        return self.encoder(self.input_adapter(batch)['features'].float())


def models(condition=True, normalize_time=True, discrete=True, **options):
    torch.manual_seed(4)
    settings = args()
    base = MixerTPP(TinyBase(settings), nn.Linear(8, 6), nn.Sequential(nn.Linear(8, 1)),
                    dropout=0., predict_b=True, loss_reduction='none', b_range=(.2, 2.))
    with torch.no_grad():
        base.hypernet_mag[0].weight.zero_()
        base.hypernet_mag[0].bias.fill_(1.)
    reference = copy.deepcopy(base)
    model = RegionalMixerTPP(base, settings, condition_regions=condition,
                             normalize_output_time=normalize_time, discrete_magnitudes=discrete, **options)
    return reference, model


def test_existing_adapter_equivalence():
    settings = args()
    batch = Batch.from_list([sequence()])
    expected = MixerBatchAdapter(settings)(batch)
    actual = RegionalBatchAdapter(settings)(batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-6, check_dtype=False)


def test_regional_wrapper_preserves_oracle_means_and_physical_time_scaling():
    settings = args()
    base = MixerTPP(TinyBase(settings), nn.Linear(8, 6), nn.Sequential(nn.Linear(8, 1)),
                    dropout=0., predict_b=True, b_range=(.2, 2.), time_parametrization='oracle')
    with torch.no_grad():
        base.hypernet_time.weight.zero_()
        base.hypernet_time.bias[:2].fill_(.12)
        base.hypernet_time.bias[2:4].fill_(math.log(math.expm1(1.)))
        base.hypernet_time.bias[4:].zero_()
    model = RegionalMixerTPP(base, settings)
    batch = Batch.from_list([sequence(tau=.1), sequence(tau=4.)])
    context = model.get_context(batch)
    distribution = model.get_inter_time_dist(context).component_distribution
    # Both shape-1 components must have Oracle's requested mean, in physical days.
    expected_mean = math.exp(math.tanh(.12) * 5 * 2.302585093)
    tau = context[..., model.context_size + 2].unsqueeze(-1)
    torch.testing.assert_close(distribution.scale.reciprocal(),
                               (expected_mean * tau).expand_as(distribution.scale), rtol=2e-6, atol=2e-6)
    assert model.inter_time_decoder.parametrization == 'oracle'


def test_legacy_likelihood_equivalence_when_new_options_disabled():
    old, new = models(condition=False, normalize_time=False, discrete=False)
    batch = Batch.from_list([sequence()])
    expected, actual = old.nll_loss(batch), new.nll_loss(batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-6, check_dtype=False)


def test_mixed_batch_equals_separate_regions_and_has_finite_gradients():
    _, model = models()
    sequences = [sequence(2., .1), sequence(3., 4., n=3, step=.01, kind=2)]
    mixed = model.nll_loss(Batch.from_list(sequences))
    for j, seq in enumerate(sequences):
        separate = model.nll_loss(Batch.from_list([seq]))
        for key in mixed:
            torch.testing.assert_close(mixed[key][j:j+1], separate[key], atol=3e-6, rtol=3e-6)
    mixed['total'].mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.conditioner[-1].weight.grad.abs().sum() > 0


def test_time_density_jacobian_and_survival_rescaling():
    _, model = models()
    context = model.get_context(Batch.from_list([sequence(tau=.1), sequence(tau=4.)]))
    physical = model.get_inter_time_dist(context)
    model.normalize_output_time = False
    unit = model.get_inter_time_dist(context)
    tau = context[..., model.context_size + 2]
    t = torch.full_like(tau, .8) * tau
    torch.testing.assert_close(physical.log_prob(t), unit.log_prob(t/tau) - tau.log(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(physical.log_survival(t), unit.log_survival(t/tau), atol=1e-6, rtol=1e-6)


def test_grid_mass_normalization_support_and_existing_convention():
    b = torch.ones(2, 81, dtype=torch.float64)
    mc = torch.tensor([[2.], [3.]], dtype=torch.float64)
    step = torch.tensor([[.1], [.1]], dtype=torch.float64)
    marks = mc + torch.arange(81, dtype=torch.float64) * step
    model = RegionalGridGR(b, mc, step)
    torch.testing.assert_close(model.log_prob(marks).exp().sum(-1), torch.ones(2, dtype=torch.float64))
    for row in range(2):
        raw = torch.full((81, 1), inverse_positive(math.log(10.)), dtype=torch.float64)
        reference = MagnitudeGridDistribution(raw, 'gr', 1, float(mc[row]), .1)
        torch.testing.assert_close(model.log_prob(marks)[row], reference.log_prob(marks[row]), atol=2e-6, rtol=2e-6)
    assert torch.isneginf(model.log_prob(mc - .1)).all()


def test_invalid_region_scale_fails_loudly():
    batch = Batch.from_list([sequence()])
    batch.region_params[..., 2] = 0
    with pytest.raises(ValueError, match='positive'):
        RegionalBatchAdapter(args())(batch)


def test_scale_equivariance_without_conditioner():
    _, model = models(condition=False)
    first = sequence(mc=2., tau=.1, step=.2)
    second = sequence(mc=3., tau=4., step=.1)
    second.region_params[:, 1] = 2.
    batch = Batch.from_list([first, second])
    context = model.get_context(batch)
    torch.testing.assert_close(context[0, :, :model.context_size], context[1, :, :model.context_size], atol=1e-6, rtol=1e-6)
    b = model._get_b_pred(context)
    torch.testing.assert_close(b[0], b[1]/2)
    loss = model.nll_loss(batch, reduction='per_event')
    torch.testing.assert_close(loss['time'][0]-math.log(.1), loss['time'][1]-math.log(4.), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(loss['mag'][0], loss['mag'][1], atol=1e-6, rtol=1e-6)


def test_future_events_cannot_change_past_context_with_fixed_metadata():
    _, model = models()
    seq = sequence()
    changed = copy.deepcopy(seq)
    changed.mag[2:] += .8
    changed.inter_times[2:] *= 2
    changed.arrival_times = changed.inter_times.cumsum(-1)[:-1]
    original_context = model.get_context(Batch.from_list([seq]))
    changed_context = model.get_context(Batch.from_list([changed]))
    torch.testing.assert_close(original_context[:, :3], changed_context[:, :3])


@pytest.mark.parametrize('mode', ['film', 'concat'])
def test_categorical_region_embeddings_are_used_and_receive_gradients(mode):
    _, model = models(num_regions=2, conditioning_mode=mode)
    with torch.no_grad():
        model.region_embedding.weight[1].fill_(1.)
        model.region_embedding.weight[2].fill_(2.)
        if mode == 'film':
            model.conditioner[-1].weight.fill_(.02)
        else:
            model.context_projection[0].weight[:, model.context_size:].fill_(.02)
    sequences = [attach_region(sequence(), [2., 1., 1., 0., 3., .1, 1], region_id=i) for i in [1,2]]
    context = model.get_context(Batch.from_list(sequences))
    assert not torch.allclose(context[0], context[1])
    model.nll_loss(Batch.from_list(sequences))['total'].sum().backward()
    assert model.region_embedding.weight.grad[1:].abs().sum() > 0
    assert torch.count_nonzero(model.region_embedding.weight.grad[0]) == 0


def test_unknown_id_fallback_and_invalid_id():
    _, model = models(num_regions=2)
    default = model.get_context(Batch.from_list([sequence()]))
    explicit = sequence()
    explicit.region_id = torch.zeros(len(explicit), dtype=torch.long)
    torch.testing.assert_close(default, model.get_context(Batch.from_list([explicit])))
    explicit.region_id.fill_(3)
    with pytest.raises(ValueError, match='vocabulary'):
        model.get_context(Batch.from_list([explicit]))


def test_new_id_initialization_and_embedding_only_update_keep_other_rows_fixed():
    _, model = models(num_regions=3)
    model.initialize_region_embedding(3, [1,2])
    torch.testing.assert_close(model.region_embedding.weight[3], model.region_embedding.weight[1:3].mean(0))
    for p in model.parameters():
        p.requires_grad_(False)
    weight = model.region_embedding.weight
    weight.requires_grad_(True)
    before = weight.detach().clone()
    mask = torch.zeros_like(weight);mask[3] = 1
    handle = weight.register_hook(lambda gradient: gradient*mask)
    optimizer = torch.optim.AdamW([weight], lr=.01, weight_decay=0.)
    optimizer.zero_grad();weight.sum().backward();optimizer.step();handle.remove()
    torch.testing.assert_close(weight[:3], before[:3])
    assert not torch.equal(weight[3], before[3])


@pytest.mark.parametrize('mode', ['film', 'concat'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual Mamba CUDA kernels')
def test_cuda_cached_generation_retains_region_for_every_trajectory(mode):
    from omegaconf import OmegaConf
    from src.models.builders import ModelBuilder
    settings = OmegaConf.load('experiments/revision/main/configs/pretrain_azdx_main_seed_0.yaml')
    for key, value in args().items():
        settings[key] = value
    settings.features_input_keys = ['mag']
    settings.use_b_updater = False
    settings.use_double_precision = False
    settings.dropout = 0.
    settings.mixer_model_config.dropout_prob = 0.
    settings.mixer_model_config.input_proj_dropout = 0.
    device = torch.device('cuda:0')
    base = ModelBuilder.by_name('mixer_tpp')()(settings, device)
    model = RegionalMixerTPP(base, settings, num_regions=3, conditioning_mode=mode).eval()
    with torch.no_grad():
        model.hypernet_time.weight.zero_()
        model.hypernet_time.bias[:2*model.num_components].fill_(math.log(math.expm1(1.)))
        model.hypernet_time.bias[2*model.num_components:].zero_()
        model.hypernet_mag[0].weight.zero_()
        model.hypernet_mag[0].bias.fill_(1.)
    past = sequence(mc=3., tau=.01, step=.1, kind=1).to(device)
    past.region_id = torch.full((len(past),), 3, device=device, dtype=torch.long)
    generated = model.sample(4, duration=.02, past_seq=past, max_sample_len=64)
    assert generated.region_params.shape[0] == 4
    assert torch.all(generated.region_params[..., 0] == 3.)
    assert torch.all(generated.region_id == 3)
    for i in range(4):
        n = int(generated.end_idx[i])
        assert (generated.mag[i, :n] >= 3.).all()
        torch.testing.assert_close(generated.inter_times[i].sum(), torch.tensor(.02, device=device, dtype=generated.inter_times.dtype))
