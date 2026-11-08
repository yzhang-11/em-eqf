"""Check continuous-time RoPE using the attention values it determines."""
import math

import pytest
import torch
from flash_attn.layers.rotary import RotaryEmbedding as RotaryEmbeddingOrig
from src.models.mha.rotary_embedding_time import RotaryEmbeddingTime


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('scale_base', [None, 1024])
def test_integer_times_match_original_rotary(scale_base):
    torch.manual_seed(42)
    qkv = torch.randn(1, 500, 3, 2, 8, device='cuda', dtype=torch.float16)
    times = torch.arange(500, device='cuda', dtype=torch.float32)[None]
    expected = RotaryEmbeddingOrig(8, interleaved=False, scale_base=scale_base).cuda()(qkv.clone(), num_heads_q=2)
    actual = RotaryEmbeddingTime(8, interleaved=False, scale_base=scale_base).cuda()(
        qkv.clone(), times=times, num_heads_q=2)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=4e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('scale_base', [None, 1024])
@pytest.mark.parametrize('irregular', [False, True])
def test_cached_attention_matches_full_time_rotary(scale_base, irregular):
    torch.manual_seed(43)
    length = 500
    qkv = torch.randn(2, length, 3, 2, 8, device='cuda', dtype=torch.float16)
    times = (torch.rand(2, length, device='cuda', dtype=torch.float64)+.2).cumsum(-1)+13.7 if irregular else (
        torch.arange(length, device='cuda', dtype=torch.float64)[None].expand(2, -1))
    full = RotaryEmbeddingTime(8, interleaved=False, scale_base=scale_base).cuda()(
        qkv.clone(), times=times, num_heads_q=2)
    rope = RotaryEmbeddingTime(8, interleaved=False, scale_base=scale_base).cuda()
    pieces = [rope(qkv[:, i:i+1].clone(), times=times[:, i:i+1], num_heads_q=2,
                   seqlen_offset=i, max_seqlen=length) for i in range(length)]
    cached = torch.cat(pieces, dim=1)
    # xPos uses reciprocal Q/K scaling. Different fixed centers change each
    # vector but cancel in every Q_i K_j product, including cached decoding.
    def attention(rotated):
        q, k, v = rotated.float().unbind(2)
        scores = torch.einsum('blhd,bshd->bhls', q, k)/math.sqrt(8)
        mask = torch.ones(length, length, device='cuda', dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, -torch.inf)
        return scores, torch.einsum('bhls,bshd->blhd', scores.softmax(-1), v)
    full_scores, full_output = attention(full)
    cache_scores, cache_output = attention(cached)
    valid = torch.isfinite(full_scores)
    torch.testing.assert_close(cache_scores[valid], full_scores[valid], rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(cache_output, full_output, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(cached[:, :, 2], qkv[:, :, 2], rtol=0, atol=0)
