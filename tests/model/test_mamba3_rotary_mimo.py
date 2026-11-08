"""MIMO hybrid phases: independent recurrence and actual native kernel paths."""
import copy
import importlib
import math

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("mamba_ssm.modules.mamba3")
from mamba_ssm import Mamba3
from mamba_ssm.utils.generation import InferenceParams
from src.models.mamba.mamba3_rotary import Mamba3Rotary

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires native MIMO GPU kernels")


def options(rank=2, **kwargs):
    result=dict(d_model=64,d_state=64,headdim=32,expand=2,is_mimo=True,mimo_rank=rank,
        chunk_size=16 if rank==2 else 8,layer_idx=0,device="cuda",dtype=torch.bfloat16)
    result.update(kwargs)
    return result


def times(length, batch=2):
    gaps=torch.where(torch.arange(length,device="cuda")%5==0,1.3,.07).double()
    return gaps.cumsum(0)[None].expand(batch,-1).contiguous()


def relative(actual, expected):
    return float(((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-8)).detach())


def randomize_mimo_weights(model):
    with torch.no_grad():
        for parameter in [model.mimo_x,model.mimo_z,model.mimo_o]:
            parameter.uniform_(.25,1.25)
    return model


def reference(model, x, event_time):
    """Explicit rank-summed recurrence; no model projection/phase/scan helpers."""
    batch,length,_=x.shape
    h,p,n,r,g=model.nheads,model.headdim,model.d_state,model.mimo_rank,model.num_bc_heads
    z,v,b,c,raw_dt,raw_a,raw_trap,raw_angle=model.in_proj(x).split(
        [model.d_inner,model.d_inner,r*g*n,r*g*n,h,h,h,model.num_rope_angles],-1)
    v=v.reshape(batch,length,h,p).to(torch.bfloat16)
    z=z.reshape(batch,length,h,p)
    dt=F.softplus(raw_dt.float()+model.dt_bias.float())
    raw_a=raw_a.float()
    a=-torch.where(raw_a>=0,1+raw_a,1/(1-raw_a.clamp_max(0))).clamp_min(model.A_floor)
    mixing=raw_trap.to(torch.bfloat16).float().sigmoid()
    native=(math.pi*raw_angle.float().tanh().unsqueeze(2)*dt.unsqueeze(-1)).cumsum(1)
    frequency=model.rotary_emb_base**(-torch.arange(model.num_rope_angles,device=x.device,dtype=torch.float64)/model.num_rope_angles)
    physical=torch.remainder(event_time[...,None]*frequency,2*math.pi).float().unsqueeze(2)
    phase=torch.remainder(native+physical,2*math.pi).unsqueeze(2)
    def rotary(raw,weight,bias):
        raw=raw.reshape(batch,length,r,g,n)
        raw=(raw.float()*torch.rsqrt(raw.float().square().mean(-1,keepdim=True)+1e-5)*weight.float()).to(torch.bfloat16)
        raw=raw.repeat_interleave(h//g,dim=3)
        raw=(raw+bias.permute(1,0,2).to(torch.bfloat16)).float()
        half=n//2
        count=model.num_rope_angles
        first=raw[...,:half]
        second=raw[...,half:]
        left=first[...,:count]*phase.cos()-second[...,:count]*phase.sin()
        right=first[...,:count]*phase.sin()+second[...,:count]*phase.cos()
        return torch.cat([left,first[...,count:],right,second[...,count:]],-1).to(torch.bfloat16).float()
    b=rotary(b,model.B_norm.weight,model.B_bias)
    c=rotary(c,model.C_norm.weight,model.C_bias)
    psi=model.mimo_x.permute(1,0,2).to(torch.bfloat16)
    projected_v=(v[:,:,None]*psi).to(torch.bfloat16).float()
    state=torch.zeros(batch,h,p,n,device=x.device)
    last_k=torch.zeros(batch,r,h,n,device=x.device)
    last_v=torch.zeros(batch,r,h,p,device=x.device)
    output=[]
    for i in range(length):
        decay=(a[:,i]*dt[:,i]).exp()
        beta=decay*dt[:,i]*(1-mixing[:,i])
        gamma=dt[:,i]*mixing[:,i]
        old_write=torch.einsum("brhp,brhn->bhpn",last_v,last_k)
        new_write=torch.einsum("brhp,brhn->bhpn",projected_v[:,i],b[:,i])
        state=decay[...,None,None]*state+beta[...,None,None]*old_write+gamma[...,None,None]*new_write
        out=torch.einsum("bhpn,brhn->brhp",state,c[:,i])
        out=out+model.D.float()[None,None,:,None]*projected_v[:,i]
        output.append(out)
        last_k,last_v=b[:,i],projected_v[:,i]
    out=torch.stack(output,1)
    z_rank=z.float().unsqueeze(2)*model.mimo_z.permute(1,0,2).float()
    if model.is_outproj_norm and not model.fuse_pregate_headwise_norm:
        out=out.to(torch.bfloat16).float()
        out=out*torch.rsqrt(out.square().mean(-1,keepdim=True)+model.norm.eps)
        out=out*model.norm.weight.float().reshape(1,1,1,h,p)
        out=out*F.silu(z_rank)
        reduced=(out*model.mimo_o.permute(1,0,2).float()).sum(2)
    else:
        if model.fuse_pregate_headwise_norm:
            out=out*torch.rsqrt(out.square().mean(-1,keepdim=True)+model.norm.eps)
            out=out*model.norm.weight.float().reshape(1,1,1,h,p)
        gate=F.silu(z_rank).to(torch.bfloat16).float()
        terms=(out*gate*model.mimo_o.permute(1,0,2).to(torch.bfloat16).float()).to(torch.bfloat16)
        reduced=torch.zeros_like(terms[:,:,0])
        for rank in range(r):
            reduced=reduced+terms[:,:,rank]
    return model.out_proj(reduced.flatten(-2).to(x.dtype))


def test_rank_two_scan_and_gradients_match_independent_recurrence():
    torch.manual_seed(61)
    model=randomize_mimo_weights(Mamba3Rotary(**options(ngroups=2)))
    check=copy.deepcopy(model)
    x=torch.randn(2,33,64,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    rx=x.detach().clone().requires_grad_(True)
    t=times(33).requires_grad_(True)
    rt=t.detach().clone().requires_grad_(True)
    actual=model(x,t)
    expected=reference(check,rx,rt)
    assert relative(actual,expected)<.035
    weights=torch.randn_like(actual).float()
    (actual.float()*weights).mean().backward()
    (expected.float()*weights).mean().backward()
    assert relative(x.grad,rx.grad)<.15
    assert relative(t.grad,rt.grad)<.15
    for (name,parameter),(other_name,other) in zip(model.named_parameters(),check.named_parameters()):
        assert name==other_name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
        assert relative(parameter.grad,other.grad)<.2,(name,relative(parameter.grad,other.grad))
    assert model.in_proj.weight.grad[-model.num_rope_angles:].abs().sum()>0


@pytest.mark.parametrize("rank,norm,fuse,groups,rope",[
    (2,False,False,2,.5),(4,True,True,1,1.),(2,True,False,1,.5),
])
def test_native_cute_decode_matches_native_tilelang_scan(monkeypatch,rank,norm,fuse,groups,rope):
    torch.manual_seed(67)
    model=randomize_mimo_weights(Mamba3Rotary(**options(rank,
        is_outproj_norm=norm,fuse_pregate_headwise_norm=fuse,ngroups=groups,rope_fraction=rope))).eval()
    module=importlib.import_module("src.models.mamba.mamba3_rotary_mimo")
    original=module.mamba3_step_fn
    calls=[]
    def native_spy(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,"mamba3_step_fn",native_spy)
    x=torch.randn(2,33,64,device="cuda",dtype=torch.bfloat16)
    t=times(33)
    cache=InferenceParams(max_seqlen=64,max_batch_size=2)
    with torch.no_grad():
        expected=model(x,t)
        pieces=[model(x[:,:29],t[:,:29],inference_params=cache)]
        cache.seqlen_offset=29
        pieces.append(model(x[:,29:32],t[:,29:32],inference_params=cache))
        cache.seqlen_offset=32
        pieces.append(model(x[:,32:],t[:,32:],inference_params=cache))
        actual=torch.cat(pieces,1)
    assert len(calls)==4
    assert torch.isfinite(actual).all()
    assert relative(actual,expected)<.035
    assert relative(actual,reference(model,x,t))<.035
    assert cache.key_value_memory_dict[0][2].shape==(2,rank,model.nheads,model.d_state)


def test_zero_time_reduces_to_native_mimo_and_irregular_times_are_used():
    torch.manual_seed(71)
    model=randomize_mimo_weights(Mamba3Rotary(**options())).eval()
    native=Mamba3(**options()).eval()
    native.load_state_dict(model.state_dict(),strict=True)
    x=torch.randn(2,33,64,device="cuda",dtype=torch.bfloat16)
    t=times(33)
    with torch.no_grad():
        zero=model(x,torch.zeros_like(t))
        original=native(x)
        physical=model(x,t)
        uniform=model(x,torch.linspace(float(t[0,0]),float(t[0,-1]),33,device=x.device,dtype=torch.float64)[None].expand(2,-1))
        shifted=model(x,t+19.3)
    assert relative(zero,original)<.01
    assert relative(physical,uniform)>.001
    assert relative(shifted,physical)<.035


def test_mimo_prefix_broadcast_and_two_dimensional_step():
    torch.manual_seed(73)
    model=Mamba3Rotary(**options()).eval()
    x=torch.randn(1,33,64,device="cuda",dtype=torch.bfloat16)
    t=times(33,batch=1)
    cache=InferenceParams(max_seqlen=64,max_batch_size=3,
        key_value_memory_dict={0:model.allocate_inference_cache(3,64)})
    with torch.no_grad():
        expected=model(x,t)[:,-1:].expand(3,-1,-1)
        model(x[:,:32],t[:,:32],inference_params=cache)
        cache.seqlen_offset=32
        output=model(x[:,32:].expand(3,-1,-1),t[:,32:].expand(3,-1),inference_params=cache)
        state=model.allocate_inference_cache(1,64)
        first=model.step(x[:,0],*state,times=t[:,:1])[0]
        reference_first=reference(model,x[:,:1],t[:,:1]).squeeze(1)
    assert relative(output,expected)<.035
    assert first.shape==(1,64)
    assert relative(first,reference_first)<.035


def test_native_cute_two_heads_alignment_and_noncontiguous_projection():
    torch.manual_seed(79)
    model=Mamba3Rotary(**options(headdim=64,chunk_size=8)).eval()
    x=torch.randn(2,1,64,device="cuda",dtype=torch.bfloat16)
    t=times(1)
    states=model.allocate_inference_cache(2,16)
    with torch.no_grad():
        actual=model.step(x,*states,times=t)[0]
        expected=reference(model,x,t)
    assert relative(actual,expected)<.035


@pytest.mark.parametrize("rank,fused",[(4,True),(2,False)])
def test_mimo_training_with_output_norm_has_finite_native_gradients(rank,fused):
    torch.manual_seed(83)
    model=randomize_mimo_weights(Mamba3Rotary(**options(rank,
        is_outproj_norm=True,fuse_pregate_headwise_norm=fused)))
    x=torch.randn(2,33,64,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    output=model(x,times(33))
    loss=output.float().square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
    assert model.in_proj.weight.grad[-model.num_rope_angles:].abs().sum()>0
    torch.optim.AdamW(model.parameters(),lr=1e-4).step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
