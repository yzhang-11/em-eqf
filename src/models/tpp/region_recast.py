"""Recurrent likelihood baselines with shared regional inputs and optional FiLM."""

import math
import torch
from torch.nn import functional as F

from .recurrent.model_v1 import RecurrentTPP
from .recurrent.model_v2 import RecurrentTPPV2
from .recurrent.utils import run_rnn_with_chunking
from .region_mixer_tpp import (
    REGIONAL_CONDITION_SIZE,
    RegionalBatchAdapter,
    apply_region_film,
    make_region_film,
    region_condition_features,
    region_params,
    to_physical_waiting_time,
)


class RegionalRecurrentInputs:
    def _init_regional_inputs(self, args, device):
        self.input_adapter = RegionalBatchAdapter(args)
        self.conditioner = make_region_film(REGIONAL_CONDITION_SIZE, self.context_size).to(device)
        # Consume the same initialization RNG for paired FiLM ablations.
        if not getattr(args, "region_condition_stats", True):
            self.conditioner = None

    def get_context(self, batch):
        adapted = self.input_adapter(batch)
        features = adapted["features"].to(self.rnn.weight_ih_l0)
        hidden = self._context_from_features(features, adapted["input_mask"])
        params = region_params(batch)
        if self.conditioner is not None:
            hidden = apply_region_film(hidden, region_condition_features(params).to(hidden), self.conditioner)
        return torch.cat((hidden, params[:, None, 2:3].expand(-1, hidden.size(1), -1)), dim=-1)

    def get_inter_time_dist(self, context):
        hidden = context[..., :-1].to(next(self.hypernet_time.parameters()))
        return to_physical_waiting_time(super().get_inter_time_dist(hidden), context[..., -1:])

    def nll_loss(self, batch, *, reduction=None, return_dict=True, eps=0.0):
        return super().nll_loss(batch, reduction=reduction, return_dict=return_dict, eps=eps)


class RegionalRECAST(RegionalRecurrentInputs, RecurrentTPP):
    def __init__(self, args, device):
        super().__init__(args, device=device)
        self._init_regional_inputs(args, device)


class RegionalRTPPV2(RegionalRecurrentInputs, RecurrentTPPV2):
    def __init__(self, args, device):
        from src.models.builders import ModelBuilder

        native = ModelBuilder.by_name("rtpp_v2")()(args, device)
        super().__init__(args, backbone=native.backbone, hypernet_time=native.hypernet_time,
                         hypernet_mag=native.hypernet_mag, device=device, bg_model=native.bg_model)
        self.rnn_chunk_len = int(args.rnn_chunk_len)
        self._init_regional_inputs(args, device)

    def _context_from_features(self, features, input_mask):
        if self.backbone.time_preprocess == "oracle":
            features = features.clone()
            time_index = self.input_adapter.features_input_keys.index("log_inter_times")
            features[..., time_index] /= math.log(10) * self.backbone.log_tau_std
        output, _ = run_rnn_with_chunking(
            rnn=self.rnn, features=features, context_size=self.context_size,
            chunk_len=self.rnn_chunk_len,
        )
        output = self.backbone._post_rnn_transform(output, features)
        output = output * input_mask[:, :, None]
        return self.dropout(F.pad(output[:, :-1, :], (0, 0, 1, 0)))
