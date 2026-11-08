"""DeepAR-style global conditioning: static region embeddings enter every token.

The probability heads, per-series scaling and inference cache are shared with
RegionalMixerTPP. No recurrent/time-encoding algorithm is changed.
"""
import torch
from torch import nn

from src.models.tpp.region_mixer_tpp import RegionalBatchAdapter, RegionalMixerTPP


class EmbeddedRegionalBatchAdapter(RegionalBatchAdapter):
    def __init__(self, args, embedding):
        super().__init__(args)
        self.embedding = embedding

    def __call__(self, batch):
        result = super().__call__(batch)
        ids = (batch.region_id[:, 0] if 'region_id' in batch else
               torch.zeros(batch.batch_size, device=batch.device, dtype=torch.long)).long()
        static = self.embedding(ids)[:, None].expand(-1, batch.seq_len, -1)
        static = static * batch.input_mask.unsqueeze(-1)
        result['features'] = torch.cat([result['features'], static], -1)
        return result


class GlobalMixerTPP(RegionalMixerTPP):
    def __init__(self, model, args, *, num_regions, region_embedding_dim=8):
        super().__init__(model, args, condition_regions=False)
        if num_regions < 1 or region_embedding_dim < 1:
            raise ValueError('Positive region vocabulary and embedding dimensions required')
        self.region_embedding = nn.Embedding(num_regions+1, region_embedding_dim,
                                            padding_idx=0).to(self.device)
        self.base_model.input_adapter = EmbeddedRegionalBatchAdapter(args, self.region_embedding)

    def _pack_context(self, hidden, params, region_ids=None):
        # Conditioning has already happened inside the shared autoregressive encoder.
        return torch.cat([hidden, params[:, None].expand(-1, hidden.shape[1], -1)], -1)
