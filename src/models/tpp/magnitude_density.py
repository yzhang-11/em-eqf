"""Normalized magnitude-grid probabilities with exponential, Weibull or Kumaraswamy components."""
import math

import torch
from torch import nn
from torch.distributions import Categorical, constraints

from src.distributions.distribution import Distribution


def positive(raw, span=8.):
    return torch.exp(span * torch.tanh(raw / span))


def inverse_positive(value, span=8.):
    return span * math.atanh(math.log(value) / span)


class MagnitudeGridDistribution(Distribution):
    """Reported mark Mc+k*step has the mass of latent excess [k*step,(k+1)*step)."""
    arg_constraints = {}
    has_rsample = False

    def __init__(self, raw, family, components, mag_min, step, mag_max=10.):
        self.raw, self.family, self.components = raw, family, components
        self.mag_min, self.step, self.mag_max = float(mag_min), float(step), float(mag_max)
        self.width = self.mag_max - self.mag_min
        self.num_bins = int(round(self.width / self.step))
        assert abs(self.num_bins * self.step - self.width) < 1e-8
        params = raw.double()
        if family == 'gr':
            self.log_weights = torch.zeros_like(params[..., :1])
            self.first = positive(params[..., :1])
            self.second = torch.ones_like(self.first)
        else:
            self.log_weights = params[..., :components].log_softmax(-1)
            self.first = positive(params[..., components:2*components], 4. if family == 'kumaraswamy' else 8.)
            self.second = positive(params[..., 2*components:3*components], 8. if family == 'kumaraswamy' else 4.)
        super().__init__(batch_shape=raw.shape[:-1], validate_args=False)

    @property
    def support(self):
        return constraints.interval(self.mag_min, self.mag_max - self.step)

    def _log_survival(self, excess):
        value = excess.double().unsqueeze(-1)
        if self.family in {'gr', 'weibull'}:
            safe = value.clamp_min(1e-15)
            power = torch.exp(self.second * safe.log())
            return torch.where(value <= 0, torch.zeros_like(power), -self.first * power)
        unit = (value / self.width).clamp(1e-15, 1. - 1e-15)
        log_power = self.first * unit.log()
        result = self.second * torch.log(-torch.expm1(log_power))
        result = torch.where(value <= 0, torch.zeros_like(result), result)
        return torch.where(value >= self.width, torch.full_like(result, -torch.inf), result)

    def _log_normalizer(self):
        edge = self.first.new_full(self.batch_shape, self.width)
        return torch.log(-torch.expm1(self._log_survival(edge)))

    def log_prob(self, magnitude):
        value = torch.as_tensor(magnitude, device=self.raw.device).double()
        index = torch.round((value - self.mag_min) / self.step)
        lower = index * self.step
        upper = (lower + self.step).clamp_max(self.width)
        valid = (index >= 0) & (index < self.num_bins)
        # Evaluate harmless endpoints for padding/out-of-support positions.
        lo = self._log_survival(lower.clamp(0, self.width - self.step))
        hi = self._log_survival(upper.clamp(self.step, self.width))
        difference = (hi - lo).clamp_max(-torch.finfo(torch.float64).tiny)
        component = lo + torch.log(-torch.expm1(difference)) - self._log_normalizer()
        result = torch.logsumexp(component + self.log_weights, -1)
        return torch.where(valid, result, torch.full_like(result, -torch.inf))

    def log_likelihood(self, magnitude, mask=None):
        values = self.log_prob(magnitude)
        if mask is not None:
            values = torch.where(mask.bool(), values, torch.zeros_like(values))
        return values.sum(-1)

    def cdf(self, magnitude):
        value = torch.as_tensor(magnitude, device=self.raw.device).double()
        index = torch.floor((value - self.mag_min) / self.step + 1e-4)
        upper = ((index + 1) * self.step).clamp(0, self.width)
        log_sf = self._log_survival(upper)
        component = -torch.expm1(log_sf) / self._log_normalizer().exp()
        result = (component * self.log_weights.exp()).sum(-1)
        result = torch.where(value < self.mag_min - 1e-6, torch.zeros_like(result), result)
        return result.clamp(0, 1)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_shape = torch.Size(sample_shape)
            component = Categorical(logits=self.log_weights).sample(sample_shape)
            shape = sample_shape + self.batch_shape
            first = self.first.expand(shape + (self.components,)).gather(-1, component.unsqueeze(-1)).squeeze(-1)
            second = self.second.expand(shape + (self.components,)).gather(-1, component.unsqueeze(-1)).squeeze(-1)
            uniform = torch.rand(shape, device=self.raw.device, dtype=torch.float64).clamp(1e-15, 1-1e-15)
            if self.family == 'kumaraswamy':
                unit = torch.exp(torch.log(-torch.expm1(torch.log1p(-uniform) / second)) / first)
                latent = self.width * unit
            else:
                normalizer = -torch.expm1(-first * self.width ** second)
                latent = torch.exp((torch.log(-torch.log1p(-uniform * normalizer)) - first.log()) / second)
            index = torch.floor(latent / self.step).clamp(0, self.num_bins - 1)
            return (self.mag_min + index * self.step).to(self.raw.dtype)


class MagnitudeDensityHead(nn.Module):
    """A likelihood-only head; no b-reference or temporal smoothness loss."""
    def __init__(self, input_dim, family, components, mag_min, step, hidden=32, mag_max=10.):
        super().__init__()
        if family not in {'gr', 'weibull', 'kumaraswamy'}:
            raise ValueError(family)
        components = 1 if family == 'gr' else int(components)
        self.family, self.components = family, components
        self.mag_min, self.step, self.mag_max = mag_min, step, mag_max
        self.input_dim, self.hidden = input_dim, hidden
        out_dim = 1 if family == 'gr' else 3 * components
        self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, out_dim)) if input_dim else nn.Linear(1, out_dim)
        last = self.network[-1] if input_dim else self.network
        nn.init.normal_(last.weight, std=.005)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            if family == 'gr':
                last.bias[0] = inverse_positive(math.log(10.))
            else:
                for index in range(components):
                    fraction = (index - .5*(components-1)) / max(1, components-1)
                    if family == 'kumaraswamy':
                        last.bias[components+index] = inverse_positive(math.exp(.4*fraction), 4.)
                        last.bias[2*components+index] = inverse_positive(math.log(10.)*(mag_max-mag_min)*math.exp(.4*fraction))
                    else:
                        last.bias[components+index] = inverse_positive(math.log(10.)*math.exp(.4*fraction))
                        last.bias[2*components+index] = inverse_positive(math.exp(.4*fraction), 4.)

    def forward(self, features):
        values = features if self.input_dim else features.new_ones(features.shape[:-1] + (1,))
        return MagnitudeGridDistribution(self.network(values), self.family, self.components,
                                         self.mag_min, self.step, self.mag_max)
