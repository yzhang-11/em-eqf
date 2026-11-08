"""PyTorch I/J-NMRP on the shared calendar input (Zhan et al., 2026)."""
import torch
from torch import nn
from torch.nn import functional as F

from .tpp_model import TPPModel


class NMRPTimeDistribution:
    def __init__(self, model, modulation):
        self.model = model
        self.modulation = modulation

    def log_prob(self, elapsed):
        hazard, cumulative = self.model.hazard_and_cumulative(elapsed, self.modulation)
        return hazard.clamp_min(1e-10).log() - cumulative

    def log_survival(self, elapsed):
        _, cumulative = self.model.hazard_and_cumulative(elapsed.clamp_min(0), self.modulation)
        return -cumulative


class NMRP(TPPModel):
    """Condition the cumulative hazard on a rolling window of past events.

    Architecture follows the archived TensorFlow model (Zenodo 18632629).
    Calendar likelihood adds the two censoring corrections to its event scores.
    """

    def __init__(self, training_gaps, training_magnitudes, *, joint, history=50, width=64):
        super().__init__()
        self.history = history
        self.joint = joint
        inputs = [torch.as_tensor(training_gaps, dtype=torch.float64)]
        if joint:
            inputs.append(torch.as_tensor(training_magnitudes, dtype=torch.float64))
        logged = torch.stack(inputs, dim=-1).clamp_min(1e-10).log()
        scale = logged.std(dim=0, unbiased=False)
        if not torch.isfinite(logged).all() or (scale <= 0).any():
            raise ValueError("Degenerate NMRP training feature statistics")
        self.register_buffer("feature_mean", logged.mean(dim=0))
        self.register_buffer("feature_scale", scale)
        self.encoder = nn.LSTM(len(inputs), width, batch_first=True)
        self.modulation = nn.Linear(width, width)
        self.time_layers = nn.ModuleList([
            nn.Linear(1, width, bias=False),
            nn.Linear(width, width, bias=False),
            nn.Linear(width, width, bias=False),
        ])
        self.readout = nn.Linear(width, 1, bias=False)
        with torch.no_grad():
            for layer in [*self.time_layers, self.readout]:
                nn.init.xavier_uniform_(layer.weight)
                layer.weight.abs_()

    def hazard_and_cumulative(self, elapsed, modulation):
        # Propagate the time derivative analytically, including at elapsed=0.
        value = elapsed.to(modulation).unsqueeze(-1)
        derivative = torch.ones_like(value)
        for layer in self.time_layers:
            weight = layer.weight.clamp_min(0)
            value = F.linear(value, weight).tanh()
            derivative = F.linear(derivative, weight) * (1 - value.square())
        weight = self.readout.weight.clamp_min(0)
        hazard = F.linear(derivative * modulation, weight).squeeze(-1)
        cumulative = F.linear(value * modulation, weight).squeeze(-1)
        return hazard.double(), cumulative.double()

    @torch.no_grad()
    def project_weights(self):
        for layer in [*self.time_layers, self.readout]:
            layer.weight.clamp_min_(0)

    def nll_loss(self, batch, *, reduction="per_event", return_dict=False):
        if batch.batch_size != 1:
            raise ValueError("Expected a single catalog calendar batch")
        first, end = int(batch.start_idx[0]), int(batch.end_idx[0])
        if first < self.history:
            raise ValueError("Supply at least one complete NMRP history window before scoring")
        gaps = batch.inter_times[0]
        features = [gaps[:end]]
        if self.joint:
            features.append(batch.mag[0, :end])
        features = torch.stack(features, dim=-1).clamp_min(1e-10).log()
        features = ((features - self.feature_mean) / self.feature_scale).to(self.modulation.weight)
        histories = features[first - self.history:].unfold(0, self.history, 1).transpose(1, 2).contiguous()
        _, (hidden, _) = self.encoder(histories)
        modulation = F.softplus(self.modulation(hidden[-1]))
        # The shared likelihood masks the unscored prefix and any padding.
        context = F.pad(modulation, (0, 0, first, gaps.numel() - end - 1), value=1).unsqueeze(0)

        def distribution(state):
            return NMRPTimeDistribution(self, state)

        nll = -self.time_log_likelihood(
            batch=batch, inter_time_dist=distribution(context), state=context,
            dist_from_state=distribution, pdf_inter_times=batch.inter_times,
            survival_inter_times=batch.inter_times,
        )
        result = self.reduce_nll_dict({"time": nll, "total": nll}, batch, reduction=reduction)
        return result if return_dict else result["total"]
