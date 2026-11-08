import pytest
import torch

from src.models.tpp.magnitude_density import MagnitudeGridDistribution
from src.models.tpp.magnitude_gated import GatedMagnitudeHead


@pytest.mark.parametrize('family', ['weibull', 'kumaraswamy'])
def test_global_shapes_and_finite_likelihood_gradients(family):
    head=GatedMagnitudeHead(3,family,2,3.,.1)
    distribution=head(torch.randn(4,3))
    torch.testing.assert_close(distribution.first,distribution.first[:1].expand(4,-1))
    torch.testing.assert_close(distribution.second,distribution.second[:1].expand(4,-1))
    loss=-distribution.log_prob(torch.tensor([3.,3.1,4.,6.])).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_temporal_probability_variation_is_bounded_by_gate_change():
    head=GatedMagnitudeHead(3,'kumaraswamy',2,3.,.1)
    distribution=head(torch.randn(2,3)*5.)
    grid=3.+torch.arange(70).double()*.1
    probability=distribution.log_prob(grid[:,None]).exp()
    pure_raw=distribution.raw.detach().double().clone()
    pure_raw[:,:2]=torch.tensor([[0.,-100.],[-100.,0.]])
    pure=MagnitudeGridDistribution(pure_raw,'kumaraswamy',2,3.,.1).log_prob(grid[:,None]).exp()
    component_distance=.5*(pure[:,0]-pure[:,1]).abs().sum()
    weight_change=(distribution.log_weights.exp()[0,0]-distribution.log_weights.exp()[1,0]).abs()
    actual=.5*(probability[:,0]-probability[:,1]).abs().sum()
    torch.testing.assert_close(actual,weight_change*component_distance,rtol=1e-8,atol=1e-12)
