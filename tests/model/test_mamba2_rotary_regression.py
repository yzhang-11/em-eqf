"""Mamba2Rotary full scan and cached inference must use identical dynamics."""
import importlib
import pytest
import torch
from mamba_ssm.utils.generation import InferenceParams
from src.models.mamba.mamba2_rotary import Mamba2Rotary

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA Mamba kernels required")


def error(actual,expected):
    return float(((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-8)).detach())


@pytest.mark.parametrize("options,fallback",[
    ({},False),
    (dict(use_conv=False,dtype=torch.bfloat16),False),
    (dict(ngroups=2,rotary_emb_scale_base=1024),False),
    (dict(use_conv=False),True),
    (dict(use_conv=False,ngroups=2),True),
    (dict(dt_limit=(.01,.02)),False),
    (dict(D_has_hdim=True),False),
    (dict(rmsnorm=False,use_conv=False),False),
])
def test_full_scan_matches_prefill_and_decode(monkeypatch,options,fallback):
    torch.manual_seed(89)
    settings=dict(d_model=64,d_state=64,headdim=32,layer_idx=0,device="cuda",dtype=torch.float32)
    settings.update(options)
    model=Mamba2Rotary(**settings).eval()
    with torch.no_grad():
        model.D.uniform_(.5,1.5)
    if fallback:
        module=importlib.import_module("src.models.mamba.mamba2_rotary")
        monkeypatch.setattr(module,"selective_state_update",None)
    x=torch.randn(2,35,64,device="cuda",dtype=settings["dtype"])
    t=torch.tensor([.1,.2,1.7,.03,.7]*7,device="cuda",dtype=torch.float64).cumsum(0)[None].expand(2,-1)+1e7
    cache=InferenceParams(max_seqlen=50,max_batch_size=2)
    with torch.no_grad():
        expected=model(x,times=t)
        pieces=[model(x[:,:32],times=t[:,:32],inference_params=cache)]
        for index in range(32,35):
            cache.seqlen_offset=index
            pieces.append(model(x[:,index:index+1],times=t[:,index:index+1],inference_params=cache))
        actual=torch.cat(pieces,1)
    assert torch.isfinite(actual).all()
    tolerance=.015 if settings["dtype"]==torch.bfloat16 else 3e-4
    assert error(actual[:,32:],expected[:,32:])<tolerance
    for state in cache.key_value_memory_dict[0]:
        if state is not None:assert torch.isfinite(state).all()


def test_real_times_forward_backward_and_optimizer():
    torch.manual_seed(97)
    model=Mamba2Rotary(d_model=64,d_state=64,headdim=32,use_conv=False,layer_idx=0,
        rotary_emb_scale_base=8192,device="cuda",dtype=torch.float32)
    x=torch.randn(2,65,64,device="cuda",requires_grad=True)
    t=torch.tensor([.01,1.3,.2,.07,.4]*13,device="cuda",dtype=torch.float64).cumsum(0)[None].expand(2,-1)
    y=model(x,times=t)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.optim.AdamW(model.parameters(),lr=1e-4).step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with torch.no_grad():
        uniform=torch.linspace(float(t[0,0]),float(t[0,-1]),65,device="cuda",dtype=torch.float64)[None].expand(2,-1)
        assert error(model(x,times=t),model(x,times=uniform))>.001
    cache=InferenceParams(max_seqlen=100,max_batch_size=2)
    model._get_states_from_cache(cache,2)
    for state in cache.key_value_memory_dict[0]:
        if state is not None:state.fill_(1)
    model._get_states_from_cache(cache,2,initialize_states=True)
    assert all(state is None or not state.any() for state in cache.key_value_memory_dict[0])
