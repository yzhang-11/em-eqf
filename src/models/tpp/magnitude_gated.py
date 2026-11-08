"""Shared magnitude components with causal history-dependent mixture weights."""
import torch
from torch import nn

from src.models.tpp.magnitude_density import MagnitudeDensityHead, MagnitudeGridDistribution


class GatedMagnitudeHead(nn.Module):
    """Only weights vary across events; component shapes are learned globally by NLL."""
    def __init__(self, input_dim, family, components, mag_min, step, hidden=32, mag_max=10.):
        super().__init__()
        if family not in {'weibull', 'kumaraswamy'} or components != 2 or input_dim < 1:
            raise ValueError((input_dim, family, components))
        self.input_dim, self.family, self.components = input_dim, family, components
        self.mag_min, self.step, self.mag_max = mag_min, step, mag_max
        initial = MagnitudeDensityHead(0, family, components, mag_min, step, mag_max=mag_max)
        self.shapes = nn.Parameter(initial.network.bias.detach()[components:].clone())
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, components))
        nn.init.normal_(self.gate[-1].weight, std=.005)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features):
        weights = self.gate(features)
        shapes = self.shapes.expand(features.shape[:-1]+self.shapes.shape)
        raw = torch.cat([weights, shapes], -1)
        return MagnitudeGridDistribution(raw, self.family, self.components,
                                         self.mag_min, self.step, self.mag_max)
