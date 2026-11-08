"""Incremental attention must retain every historical key/value."""
import importlib

import pytest
import torch
from torch.nn import functional as F
from mamba_ssm.utils.generation import InferenceParams

from src.models.mha.mha_time import MHATime


@pytest.mark.parametrize('flash_available',[False,True])
@pytest.mark.parametrize('chunk_size',[1,2])
def test_float32_decoding_matches_full_causal_attention(monkeypatch,flash_available,chunk_size):
    module=importlib.import_module('src.models.mha.mha_time')
    def reject_float32_flash(*args,**kwargs):
        raise AssertionError('Float32 cache must not be cast to a temporary flash cache')
    monkeypatch.setattr(module,'flash_attn_with_kvcache',reject_float32_flash if flash_available else None)
    torch.manual_seed(9)
    model=MHATime(embed_dim=8,num_heads=2,layer_idx=0,causal=True)
    query=torch.randn(2,4,2,4)
    kv=torch.randn(2,4,2,2,4)
    expected=F.scaled_dot_product_attention(query.transpose(1,2),
        kv[:,:,0].transpose(1,2),kv[:,:,1].transpose(1,2),is_causal=True).transpose(1,2)
    params=InferenceParams(max_seqlen=16,max_batch_size=2)
    params.key_value_memory_dict[0]=model.allocate_inference_cache(2,16)
    result=[model._update_kvcache_attention(query[:,:2],kv[:,:2],params)]
    for position in range(2,4,chunk_size):
        params.seqlen_offset=position
        result.append(model._update_kvcache_attention(query[:,position:position+chunk_size],kv[:,position:position+chunk_size],params))
    torch.testing.assert_close(torch.cat(result,dim=1),expected,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(params.key_value_memory_dict[0][0][:,:4],kv,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA attention kernels')
@pytest.mark.parametrize('dtype,cache_dtype',[
    (torch.float32,torch.float32),(torch.float16,torch.float16),
    (torch.float16,torch.float32),(torch.bfloat16,torch.bfloat16),
    (torch.bfloat16,torch.float32),
])
def test_cuda_decoding_persists_native_and_mixed_dtype_cache(dtype,cache_dtype):
    torch.manual_seed(11)
    model=MHATime(embed_dim=16,num_heads=2,layer_idx=0,causal=True).to(device='cuda',dtype=dtype)
    query=torch.randn(2,4,2,8,device='cuda',dtype=dtype)
    kv=torch.randn(2,4,2,2,8,device='cuda',dtype=dtype)
    expected=F.scaled_dot_product_attention(query.transpose(1,2),kv[:,:,0].transpose(1,2),
        kv[:,:,1].transpose(1,2),is_causal=True).transpose(1,2)
    params=InferenceParams(max_seqlen=16,max_batch_size=2)
    params.key_value_memory_dict[0]=model.allocate_inference_cache(2,16,dtype=cache_dtype)
    params.key_value_memory_dict[0][0].fill_(float('nan'))
    result=[model._update_kvcache_attention(query[:,:2],kv[:,:2],params)]
    for position in [2,3]:
        params.seqlen_offset=position
        result.append(model._update_kvcache_attention(query[:,position:position+1],kv[:,position:position+1],params))
    actual=torch.cat(result,dim=1)
    assert torch.isfinite(actual).all()
    tolerance=1e-5 if dtype==torch.float32 else .02
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(params.key_value_memory_dict[0][0][:,:4],kv.to(cache_dtype),rtol=0,atol=0)
