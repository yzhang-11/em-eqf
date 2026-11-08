import math
import pytest
import torch

from src.models.tpp.magnitude_density import MagnitudeDensityHead, MagnitudeGridDistribution, inverse_positive


@pytest.mark.parametrize('family,components', [('gr', 1), ('weibull', 1), ('weibull', 2), ('kumaraswamy', 1), ('kumaraswamy', 2)])
def test_normalization_and_finite_gradients(family, components):
    head = MagnitudeDensityHead(3, family, components, 3., .1)
    features = torch.randn(4, 3)
    distribution = head(features)
    grid = (3. + torch.arange(70).double()*.1).unsqueeze(-1).expand(70, 4)
    probabilities = distribution.log_prob(grid).exp()
    torch.testing.assert_close(probabilities.sum(0), torch.ones(4, dtype=torch.float64), atol=2e-10, rtol=2e-10)
    loss = -distribution.log_prob(torch.tensor([3., 3.1, 6., 9.9])).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
    assert torch.all(distribution.cdf(grid)[1:] >= distribution.cdf(grid)[:-1])


def test_exponential_mass_matches_geometric():
    rate = 2.3
    raw = torch.tensor([[inverse_positive(rate)]], dtype=torch.float64)
    dist = MagnitudeGridDistribution(raw, 'gr', 1, 3., .1, 4.)
    values = 3. + torch.arange(10).double()*.1
    expected = -rate * torch.arange(10).double()*.1 + math.log(-math.expm1(-rate*.1)) - math.log(-math.expm1(-rate))
    torch.testing.assert_close(dist.log_prob(values), expected)


@pytest.mark.parametrize('family', ['weibull', 'kumaraswamy'])
def test_sample_matches_reported_grid_probabilities(family):
    torch.manual_seed(41)
    head = MagnitudeDensityHead(0, family, 2, 3., .1, mag_max=4.)
    dist = head(torch.zeros(1, 1))
    sample = dist.sample((40000,)).squeeze(-1)
    index = ((sample-3.)/.1).round().long()
    assert index.min() >= 0 and index.max() < 10
    observed = torch.bincount(index, minlength=10).double() / len(index)
    expected = dist.log_prob(3. + torch.arange(10).double()*.1).exp()
    torch.testing.assert_close(observed, expected, rtol=0, atol=.012)


def test_mask_and_support():
    dist = MagnitudeDensityHead(2, 'kumaraswamy', 2, 3., .1)(torch.zeros(2, 3, 2))
    marks = torch.tensor([[3., 3.1, 0.], [3.5, 0., 0.]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert torch.isfinite(dist.log_likelihood(marks, mask)).all()
    assert torch.isneginf(dist.log_prob(marks)[mask == 0]).all()
