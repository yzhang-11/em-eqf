"""Likelihood-trained Markov states for magnitude densities, without per-event b predictions."""
import math
import torch
from torch import nn

from src.models.tpp.magnitude_density import MagnitudeGridDistribution, inverse_positive


def markov_predictive_weights(log_emission, transition, initial):
    """Parallel causal prefix scan; weight[i] excludes mark i and every future mark."""
    emissions = torch.exp(log_emission - log_emission.max(-1, keepdim=True).values)
    matrices = transition.T.unsqueeze(0) * emissions.unsqueeze(-2)
    matrices = matrices / matrices.sum((-1, -2), keepdim=True)
    offset = 1
    while offset < len(matrices):
        products = torch.bmm(matrices[offset:], matrices[:-offset])
        products = products / products.sum((-1, -2), keepdim=True)
        matrices = torch.cat([matrices[:offset], products], dim=0)
        offset *= 2
    following = (matrices @ initial.unsqueeze(-1)).squeeze(-1)
    following = following / following.sum(-1, keepdim=True)
    return torch.cat([initial.unsqueeze(0), following[:-1]], dim=0)


class MagnitudeStateModel(nn.Module):
    def __init__(self, states, mag_min, step, persistence=.99, mag_max=10.):
        super().__init__()
        self.states, self.mag_min, self.step, self.mag_max = states, mag_min, step, mag_max
        raw = torch.zeros(states, 3, dtype=torch.float64)
        for i in range(states):
            fraction = (i-.5*(states-1))/max(1,states-1)
            raw[i,1] = inverse_positive(math.exp(.6*fraction),4.)
            raw[i,2] = inverse_positive(math.log(10.)*(mag_max-mag_min)*math.exp(1.0*fraction))
        self.emission_parameters = nn.Parameter(raw)
        transition = torch.full((states,states),(1-persistence)/(states-1),dtype=torch.float64)
        transition.fill_diagonal_(persistence)
        self.transition_logits = nn.Parameter(transition.log())
        self.initial_logits = nn.Parameter(torch.zeros(states,dtype=torch.float64))

    def emissions(self, marks):
        dist=MagnitudeGridDistribution(self.emission_parameters,'kumaraswamy',1,self.mag_min,self.step,self.mag_max)
        return dist.log_prob(marks.double().unsqueeze(-1))

    def predictive_weights(self, marks):
        log_emission=self.emissions(marks)
        prior=markov_predictive_weights(log_emission,self.transition_logits.softmax(-1),self.initial_logits.softmax(-1))
        return prior,log_emission

    def log_prob_sequence(self, marks):
        prior,log_emission=self.predictive_weights(marks)
        return torch.logsumexp(prior.clamp_min(1e-300).log()+log_emission,-1)

    def distribution_sequence(self, marks):
        prior,_=self.predictive_weights(marks)
        raw=torch.cat([prior.clamp_min(1e-300).log(),
                       self.emission_parameters[:,1].expand(len(marks),-1),
                       self.emission_parameters[:,2].expand(len(marks),-1)],dim=-1)
        return MagnitudeGridDistribution(raw,'kumaraswamy',self.states,self.mag_min,self.step,self.mag_max)
