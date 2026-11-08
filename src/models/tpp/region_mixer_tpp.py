"""Opt-in multi-region MixerTPP; legacy models and time encoders are unchanged.

region_params is constant within each sequence and uses REGION_FIELDS order.
Repeat it over observed events so existing Sequence/Batch slicing can carry it.
During generation it is retained in the explicit inference cache.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import src.distributions as dist
from src.models.adapters.mixer_tpp import MixerBatchAdapter
from src.models.tpp.mixer_tpp import MixerTPP

REGION_FIELDS = ('mc', 'b_ref', 'tau_mean', 'tau_min', 'tau_max', 'mag_step', 'mag_type')
# Fixed physical measurement categories, not learned region IDs.
MAG_TYPES = {'unknown': 0, 'ML': 1, 'MJ': 2, 'Mw': 3}
REGIONAL_CONDITION_SIZE = 4 + len(MAG_TYPES)


def region_condition_features(params):
    values = torch.stack([params[:, 0], params[:, 1].log(), params[:, 2].log(), params[:, 5].log()], -1)
    kind = F.one_hot(params[:, 6].long(), len(MAG_TYPES)).to(values)
    return torch.cat([values, kind], -1)


def make_region_film(condition_size, context_size):
    conditioner = nn.Sequential(nn.Linear(condition_size, 32), nn.SiLU(),
                                nn.Linear(32, 2 * context_size))
    nn.init.zeros_(conditioner[-1].weight)
    nn.init.zeros_(conditioner[-1].bias)
    return conditioner


def apply_region_film(hidden, condition, conditioner):
    scale, offset = conditioner(condition).chunk(2, -1)
    return hidden * (1 + scale[:, None]) + offset[:, None]


def to_physical_waiting_time(distribution, tau):
    component = distribution.component_distribution
    # Project Weibull uses S(t)=exp(-scale*t**shape).
    rate = torch.exp(component.scale.log() - component.shape * tau.log())
    return dist.MixtureSameFamily(distribution.mixture_distribution,
                                  dist.Weibull(rate, component.shape))


def region_params(batch):
    values = batch.region_params[:, 0, :]
    if values.shape[-1] != len(REGION_FIELDS) or not torch.isfinite(values).all():
        raise ValueError(f'Expected finite region metadata in order {REGION_FIELDS}')
    if (values[:, [1, 2, 5]] <= 0).any() or (values[:, 3] < 0).any() or (values[:, 4] <= values[:, 3]).any():
        raise ValueError('Regional b, time scale and magnitude step must be positive; tau_max > tau_min >= 0')
    kinds = values[:, 6]
    if ((kinds != kinds.round()) | (kinds < 0) | (kinds >= len(MAG_TYPES))).any():
        raise ValueError('Unknown magnitude measurement category')
    return values


def attach_region(sequence, values, region_id=None):
    """Attach training/past-only statistics; do not estimate them from targets."""
    if len(sequence) == 0:
        raise ValueError('Regional sequences require at least one observed event')
    values = torch.as_tensor(values, device=sequence.device, dtype=torch.float64)
    if values.shape != (len(REGION_FIELDS),):
        raise ValueError(f'Expected {len(REGION_FIELDS)} region fields')
    sequence.region_params = values.expand(len(sequence), -1).clone()
    if region_id is not None:
        if int(region_id) != region_id or region_id < 0:
            raise ValueError('Region ID must be a nonnegative integer; 0 is unknown')
        sequence.region_id = torch.full((len(sequence),), int(region_id),
                                       device=sequence.device, dtype=torch.long)
    return sequence


class RegionalBatchAdapter(MixerBatchAdapter):
    def __init__(self, args):
        super().__init__(args)
        self.magnitude_preprocess = getattr(args, "region_magnitude_preprocess", "gr")
        if self.magnitude_preprocess not in ("gr", "center"):
            raise ValueError("Expected region_magnitude_preprocess to be gr or center")

    def __call__(self, batch):
        p = region_params(batch)
        mc, b, tau, low, high = [p[:, j:j+1] for j in range(5)]
        mask = batch.input_mask
        # This is exactly the existing GR standardization, now per sequence.
        feature_map = {
            'mag': ((batch.mag - mc) * b - 1 / math.log(10)).unsqueeze(-1),
            'log_inter_times': (batch.inter_times.clamp_min(self.eps).log() - tau.log()).unsqueeze(-1),
        }
        if self.magnitude_preprocess == "center":
            feature_map['mag'] = batch.mag.unsqueeze(-1) - batch.region_mag_mean
        if 'loc' in self.features_input_keys:
            feature_map['loc'] = batch.loc.unsqueeze(-1)
        features = torch.cat([feature_map[k] for k in self.features_input_keys], -1)
        features = features * mask.unsqueeze(-1)
        if self.mask_input_features:
            features = features.clone()
            for key in self.mask_input_features:
                features[..., self.features_input_keys.index(key)] = 0
        result = {'features': features, 'input_mask': mask}
        if self.provide_physical_dt:
            result['physical_dt'] = torch.where(mask.bool(), batch.inter_times / tau,
                                                torch.ones_like(batch.inter_times))
        if 'times' in self.extra_input_keys:
            times = batch.arrival_times
            if self.shift_time_to_zero:
                first = times.ne(0).long().argmax(-1)
                times = times - times.gather(1, first[:, None])
            result['times'] = (times / tau if self.normalize_time else times) * mask
        if 'inter_times' in self.extra_input_keys:
            result['inter_times'] = (batch.inter_times - low) / (high - low + self.eps) * mask
        return result


class RegionalGridGR(dist.GutenbergRichter):
    """Existing lower-edge magnitude-bin convention, with per-row Mc and step."""
    def __init__(self, b, mag_min, step, mag_max=10.):
        super().__init__(b, mag_min=mag_min, mag_max=mag_max)
        self.step = step

    def log_prob(self, x, mask=None):
        x = x.to(self.b)
        index = ((x - self.mag_min) / self.step).round()
        lower = self.mag_min + index * self.step
        width = self.mag_max - self.mag_min
        bin_width = torch.minimum(self.step, self.mag_max - lower).clamp_min(1e-15)
        rate = self.b * math.log(10)
        mass = (-rate * (lower - self.mag_min)
                + torch.log(-torch.expm1(-rate * bin_width))
                - torch.log(-torch.expm1(-rate * width)))
        valid = (index >= 0) & (lower < self.mag_max) & ((x - lower).abs() < 1e-5)
        result = torch.where(valid, mass, torch.full_like(mass, -torch.inf))
        return torch.where(mask.bool(), result, torch.zeros_like(result)) if mask is not None else result

    def cdf(self, x, b=None):
        x = torch.as_tensor(x, device=self.b.device, dtype=self.b.dtype)
        upper = self.mag_min + (torch.floor((x - self.mag_min) / self.step + 1e-6) + 1) * self.step
        return super().cdf(upper, b=b)

    def rsample(self, sample_shape=torch.Size()):
        latent = super().rsample(sample_shape)
        index = torch.floor((latent - self.mag_min) / self.step)
        return self.mag_min + index * self.step


class RegionalMixerTPP(MixerTPP):
    def __init__(self, model, args, *, condition_regions=True,
                 normalize_output_time=True, discrete_magnitudes=True, relative_b=True,
                 num_regions=0, region_embedding_dim=8, condition_stats=True,
                 conditioning_mode='film'):
        if model.b_filter is not None or model.bg_model is not None or model.use_b_updater:
            raise ValueError('Regional model currently uses likelihood-only heads without b filters or background models')
        super().__init__(model.base_model, model.hypernet_time, model.hypernet_mag,
                         dropout=model.dropout.p, predict_b=model.predict_b,
                         loss_weights=dict(model.weights), loss_reduction=model.reduction,
                         b_range=(model.b_min, model.b_max),
                         time_parametrization=model.inter_time_decoder.parametrization)
        self.base_model.input_adapter = RegionalBatchAdapter(args)
        self.normalize_output_time = normalize_output_time
        self.discrete_magnitudes = discrete_magnitudes
        self.relative_b = relative_b
        self.condition_stats = condition_stats
        self.region_embedding = None
        self.context_projection = None
        self.conditioner = None
        if conditioning_mode not in {'film', 'concat'}:
            raise ValueError('conditioning_mode must be film or concat')
        self.conditioning_mode = conditioning_mode
        condition_size = REGIONAL_CONDITION_SIZE if condition_stats else 0
        if num_regions:
            if not condition_regions or num_regions < 1 or region_embedding_dim < 1:
                raise ValueError('Region embeddings require an enabled conditioner and positive dimensions')
            # Row 0 is an explicit, zero shared fallback; IDs are categorical.
            self.region_embedding = nn.Embedding(num_regions + 1, region_embedding_dim,
                                                padding_idx=0).to(self.device)
            condition_size += region_embedding_dim
        if condition_regions:
            if not condition_size:
                raise ValueError('An enabled conditioner needs statistics or region embeddings')
            if conditioning_mode == 'film':
                self.conditioner = make_region_film(condition_size, self.context_size).to(self.device)
            else:
                # Shared nonlinear prediction head over [history, conditions].
                projection = nn.Linear(self.context_size + condition_size, self.context_size).to(self.device)
                with torch.no_grad():
                    projection.weight.zero_()
                    projection.weight[:, :self.context_size].copy_(torch.eye(self.context_size, device=self.device))
                    projection.bias.zero_()
                self.context_projection = nn.Sequential(projection, nn.SiLU())

    def _region_ids(self, batch):
        ids = (batch.region_id[:, 0] if 'region_id' in batch else
               torch.zeros(batch.batch_size, device=batch.device, dtype=torch.long))
        if self.region_embedding is not None:
            if ((ids != ids.long()) | (ids < 0) | (ids >= self.region_embedding.num_embeddings)).any():
                raise ValueError('Region ID is outside the saved embedding vocabulary')
        return ids.long()

    @torch.no_grad()
    def initialize_region_embedding(self, region_id, source_ids):
        """Initialize a new/held-out ID from trained source vectors before adaptation."""
        if self.region_embedding is None or region_id <= 0:
            raise ValueError('A positive target ID and a region embedding are required')
        if not source_ids or region_id in source_ids or 0 in source_ids:
            raise ValueError('Supply trained, positive source IDs distinct from the target')
        self.region_embedding.weight[region_id].copy_(self.region_embedding.weight[source_ids].mean(0))

    def _pack_context(self, hidden, params, region_ids=None):
        conditions = []
        if self.condition_stats and (self.conditioner is not None or self.context_projection is not None):
            conditions.append(region_condition_features(params).to(hidden))
        if self.region_embedding is not None:
            conditions.append(self.region_embedding(region_ids))
        c = torch.cat(conditions, -1) if conditions else None
        if self.conditioner is not None:
            hidden = apply_region_film(hidden, c, self.conditioner)
        if self.context_projection is not None:
            hidden = self.context_projection(torch.cat([hidden, c[:, None].expand(-1, hidden.shape[1], -1)], -1))
        # Preserve original metadata precision; heads cast only the hidden part.
        return torch.cat([hidden, params[:, None].expand(-1, hidden.shape[1], -1)], -1)

    def _hidden(self, context):
        return context[..., :self.context_size].to(self.hypernet_time.weight)

    def get_context(self, batch, inference_params=None):
        params = region_params(batch)
        return self._pack_context(super().get_context(batch, inference_params), params, self._region_ids(batch))

    def get_current_state(self, batch, inference_params=None, return_all=False):
        if inference_params is not None:
            if not hasattr(inference_params, 'region_params'):
                inference_params.region_params = region_params(batch).clone()
                inference_params.region_id = self._region_ids(batch).clone()
            params = inference_params.region_params.expand(batch.batch_size, -1)
            batch.region_params = params[:, None].expand(-1, batch.seq_len, -1)
            ids = inference_params.region_id.expand(batch.batch_size)
            batch.region_id = ids[:, None].expand(-1, batch.seq_len)
        else:
            params = region_params(batch)
            ids = self._region_ids(batch)
        hidden = super().get_current_state(batch, inference_params, return_all=return_all)
        return self._pack_context(hidden, params, ids)

    def get_inter_time_dist(self, context):
        distribution = super().get_inter_time_dist(self._hidden(context))
        if not self.normalize_output_time:
            return distribution
        tau = context[..., self.context_size + 2].unsqueeze(-1)
        return to_physical_waiting_time(distribution, tau)

    def _get_b_pred(self, context, predict_b=None, inter_times=None, filter_params=None):
        if filter_params is not None:
            raise ValueError('Regional model does not use b-filter state')
        if not (self.predict_b if predict_b is None else predict_b):
            return context[..., self.context_size + 1]
        prediction = super()._get_b_pred(self._hidden(context), predict_b=True).to(context.dtype)
        if self.relative_b:
            prediction = prediction * context[..., self.context_size + 1]
            clipped = prediction.clamp(self.b_min, self.b_max)
            prediction = prediction + (clipped - prediction).detach()
        return prediction

    def get_magnitude_dist(self, context, predict_b=None, return_b=False,
                           inter_times=None, filter_params=None):
        if filter_params is not None:
            raise ValueError('Regional model does not use b-filter state')
        b = self._get_b_pred(context, predict_b)
        mc = context[..., self.context_size]
        if self.discrete_magnitudes:
            distribution = RegionalGridGR(b, mc, context[..., self.context_size + 5])
        else:
            distribution = dist.GutenbergRichter(b, mag_min=mc)
        return (distribution, b) if return_b else distribution

    @torch.inference_mode()
    def sample(self, batch_size, duration, t_start=0., past_seq=None,
               return_sequences=False, predict_b=None, max_sample_len=1000):
        if past_seq is None or len(past_seq) == 0:
            raise ValueError('Regional sampling requires a nonempty, metadata-tagged history')
        params = past_seq.region_params[0]
        result = super().sample(batch_size, duration, t_start, past_seq,
                                return_sequences=False, predict_b=predict_b,
                                max_sample_len=max_sample_len)
        result.region_params = params.to(result.device).expand(batch_size, result.seq_len, -1).clone()
        if 'region_id' in past_seq:
            result.region_id = torch.full((batch_size, result.seq_len), int(past_seq.region_id[0]),
                                          device=result.device, dtype=torch.long)
        return result.to_list() if return_sequences else result
