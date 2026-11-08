"""Hybrid Mamba-3 phases must match an independent trapezoidal recurrence."""
import copy
import math

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("mamba_ssm.modules.mamba3")
from mamba_ssm.utils.generation import InferenceParams
from src.models.mamba.mamba3_rotary import Mamba3Rotary
from src.models.mamba.mixer_seq import MixerModel

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Mamba-3 Triton kernels")


def layer(**kwargs):
    options = dict(d_model=64, d_state=64, expand=2, headdim=64, chunk_size=64, layer_idx=0)
    options.update(kwargs)
    return Mamba3Rotary(**options)


def event_times(length, batch=2, device="cuda"):
    index = torch.arange(length, device=device)
    gaps = torch.where(index % 7 == 0, 1.7, .03).double()
    return gaps.cumsum(0).unsqueeze(0).expand(batch, -1).contiguous()


def relative_error(actual, expected):
    return float(((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-8)).detach())


def reference(model, u, times):
    """Direct Mamba-3 recurrence; no scan, step, phase or projection helpers."""
    batch, length, _ = u.shape
    z, x, b, c, dd_dt, dd_a, trap, angle = model.in_proj(u).split(
        [model.d_inner, model.d_inner, model.num_bc_heads*model.d_state,
         model.num_bc_heads*model.d_state, model.nheads, model.nheads, model.nheads, model.num_rope_angles], -1)
    x = x.reshape(batch, length, model.nheads, model.headdim).to(torch.bfloat16).float()
    z = z.reshape_as(x)
    dt = F.softplus(dd_dt.float()+model.dt_bias.float())
    a_input = dd_a.float()
    a = -torch.where(a_input >= 0, 1+a_input, 1/(1-a_input.clamp_max(0))).clamp_min(model.A_floor)
    trap = trap.to(torch.bfloat16).float().sigmoid()
    freq = model.rotary_emb_base ** (-torch.arange(model.num_rope_angles,device=u.device,dtype=torch.float64)/model.num_rope_angles)
    native_phase = (math.pi * angle.float().tanh().unsqueeze(2) * dt.unsqueeze(-1)).cumsum(1)
    physical_phase = torch.remainder(times.double()[...,None]*freq,2*math.pi).float().unsqueeze(2)
    phase = torch.remainder(native_phase + physical_phase,2*math.pi)
    def rotate(raw, weight, bias):
        raw = raw.reshape(batch,length,model.num_bc_heads,model.d_state)
        normed = (raw.float()*torch.rsqrt(raw.float().square().mean(-1,keepdim=True)+1e-5)*weight.float()).to(raw.dtype)
        normed = normed.to(torch.bfloat16).float()
        normed = normed.repeat_interleave(model.nheads//model.num_bc_heads,dim=2)+bias.squeeze(1).float()
        pairs = normed[...,:model.split_tensor_size].reshape(batch,length,model.nheads,-1,2)
        even,odd = pairs.unbind(-1)
        rotated = torch.stack((even*phase.cos()-odd*phase.sin(),even*phase.sin()+odd*phase.cos()),-1).flatten(-2)
        return torch.cat((rotated,normed[...,model.split_tensor_size:]),-1)
    b = rotate(b,model.B_norm.weight,model.B_bias)
    c = rotate(c,model.C_norm.weight,model.C_bias)
    state=torch.zeros(batch,model.nheads,model.headdim,model.d_state,device=u.device)
    prev_k=torch.zeros_like(b[:,0])
    prev_v=torch.zeros_like(x[:,0])
    outputs=[]
    for i in range(length):
        decay=(a[:,i]*dt[:,i]).exp()
        beta=decay*dt[:,i]*(1-trap[:,i])
        gamma=dt[:,i]*trap[:,i]
        state=(
            decay[...,None,None]*state
            + beta[...,None,None]*prev_v[...,None]*prev_k[...,None,:]
            + gamma[...,None,None]*x[:,i,:, :,None]*b[:,i,:,None,:]
        )
        y=(state*c[:,i,:,None,:]).sum(-1)+model.D.float()[None,:,None]*x[:,i]
        if not model.is_outproj_norm:
            y=y*F.silu(z[:,i].float())
        outputs.append(y)
        prev_k,prev_v=b[:,i],x[:,i]
    y=torch.stack(outputs,1).to(torch.bfloat16).flatten(-2)
    if model.is_outproj_norm:
        y=model.norm(y,z.flatten(-2))
    return model.out_proj(y.to(u.dtype))


def test_phases_use_real_time_and_keep_double_precision_with_bfloat16_weights():
    model=layer(dtype=torch.bfloat16)
    times=torch.tensor([[1e9+.001,1e9+.101,1e9+.701]],dtype=torch.float64)
    phase=model._time_angles(times,1,3,torch.device("cpu"))
    frequency=10000. ** (-torch.arange(model.num_rope_angles,dtype=torch.float64)/model.num_rope_angles)
    expected=torch.remainder(times[...,None]*frequency,2*math.pi).float()
    torch.testing.assert_close(phase[:,:,0],expected,rtol=0,atol=0)
    assert not torch.equal(phase[:,0],phase[:,1])
    with pytest.raises(ValueError,match="requires actual event times"):
        model._time_angles(None,1,3,torch.device("cpu"))
    with pytest.raises(ValueError,match="times must have shape"):
        model._time_angles(times[:,:1],1,3,torch.device("cpu"))


def test_unsupported_modes_fail_explicitly():
    with pytest.raises(ValueError,match="split-half"):
        layer(is_mimo=True,rotary_emb_interleaved=True)
    with pytest.raises(NotImplementedError,match="xPos"):
        layer(rotary_emb_scale_base=1024)


@CUDA
def test_scan_and_gradients_match_independent_event_time_recurrence():
    torch.manual_seed(16)
    model=layer(device="cuda",dtype=torch.bfloat16)
    comparison=copy.deepcopy(model)
    x=torch.randn(2,65,64,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    rx=x.detach().clone().requires_grad_(True)
    times=event_times(65).requires_grad_(True)
    rt=times.detach().clone().requires_grad_(True)
    actual=model(x,times)
    expected=reference(comparison,rx,rt)
    assert relative_error(actual,expected)<.025
    weights=torch.randn_like(actual).float()
    (actual.float()*weights).mean().backward()
    (expected.float()*weights).mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(times.grad).all()
    assert relative_error(x.grad,rx.grad)<.12
    assert relative_error(times.grad,rt.grad)<.12
    for (name,parameter),(rname,other) in zip(model.named_parameters(),comparison.named_parameters()):
        assert name==rname
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
        assert relative_error(parameter.grad,other.grad)<.15,(name,relative_error(parameter.grad,other.grad))
    # Native angle rows remain trainable in the additive variant.
    assert model.in_proj.out_features==2*model.d_inner+2*model.num_bc_heads*model.d_state+3*model.nheads+model.num_rope_angles
    assert model.in_proj.weight.grad[-model.num_rope_angles:].abs().sum()>0


@CUDA
@pytest.mark.parametrize("outproj_norm,rope_fraction",[(False,.5),(True,1.)])
def test_prefill_chunk_and_single_event_decode_match_full_scan(outproj_norm,rope_fraction):
    torch.manual_seed(23)
    model=layer(device="cuda",dtype=torch.bfloat16,is_outproj_norm=outproj_norm,rope_fraction=rope_fraction).eval()
    x=torch.randn(2,65,64,device="cuda",dtype=torch.bfloat16)
    times=event_times(65)
    params=InferenceParams(max_seqlen=100,max_batch_size=2)
    with torch.no_grad():
        expected=model(x,times)
        pieces=[model(x[:,:59],times[:,:59],inference_params=params)]
        params.seqlen_offset=59
        pieces.append(model(x[:,59:63],times[:,59:63],inference_params=params))
        for i in [63,64]:
            params.seqlen_offset=i
            pieces.append(model(x[:,i:i+1],times[:,i:i+1],inference_params=params))
        actual=torch.cat(pieces,1)
    assert relative_error(actual,expected)<.025
    assert torch.isfinite(actual).all()
    with torch.no_grad():
        _,_,_,_,_,dt,_,angle=model._project(x)
        native_final=(math.pi*angle.float().tanh().unsqueeze(2)*dt.unsqueeze(-1)).sum(1)
        native_final=torch.remainder(native_final,2*math.pi)
    torch.testing.assert_close(params.key_value_memory_dict[0][0],native_final,rtol=2e-5,atol=2e-5)


@CUDA
def test_single_prefix_broadcasts_all_states_to_sampling_batch():
    torch.manual_seed(29)
    model=layer(device="cuda",dtype=torch.bfloat16).eval()
    x=torch.randn(1,65,64,device="cuda",dtype=torch.bfloat16)
    times=event_times(65,batch=1)
    params=InferenceParams(max_seqlen=100,max_batch_size=3,
        key_value_memory_dict={0:model.allocate_inference_cache(3,100)})
    with torch.no_grad():
        expected=model(x,times)[:,-1:].expand(3,-1,-1)
        model(x[:,:64],times[:,:64],inference_params=params)
        params.seqlen_offset=64
        result=model(x[:,64:].expand(3,-1,-1),times[:,64:].expand(3,-1),inference_params=params)
    assert relative_error(result,expected)<.025
    states=model.allocate_inference_cache(1,100)
    with torch.no_grad():
        first=model.step(x[:,0],*states,times=times[:,:1])[0]
        full_first=model(x[:,:1],times[:,:1]).squeeze(1)
    assert first.shape==(1,64)
    assert relative_error(first,full_first)<.025


@CUDA
def test_model_registration_and_real_time_sensitivity():
    torch.manual_seed(31)
    model=MixerModel(d_model=64,n_layer=1,d_intermediate=0,input_dim=2,
        ssm_cfg=dict(layer="Mamba3Rotary",d_state=64,headdim=64),
        device="cuda",dtype=torch.bfloat16).eval()
    features=torch.randn(2,65,2,device="cuda",dtype=torch.bfloat16)
    times=event_times(65)
    uniform=torch.linspace(float(times[0,0]),float(times[0,-1]),65,device="cuda",dtype=torch.float64)[None].expand(2,-1)
    with torch.no_grad():
        irregular=model(features=features,times=times)
        regular=model(features=features,times=uniform)
        shifted=model(features=features,times=times+13.7)
    assert relative_error(irregular,regular)>.001
    assert relative_error(shifted,irregular)<.025


@CUDA
def test_zero_event_time_reduces_to_native_mamba3_and_keeps_angle_weights():
    from mamba_ssm import Mamba3
    torch.manual_seed(37)
    model=layer(device="cuda",dtype=torch.bfloat16).eval()
    native=Mamba3(d_model=64,d_state=64,expand=2,headdim=64,
        chunk_size=64,layer_idx=0,device="cuda",dtype=torch.bfloat16).eval()
    native.load_state_dict(model.state_dict(),strict=True)
    x=torch.randn(2,65,64,device="cuda",dtype=torch.bfloat16)
    with torch.no_grad():
        actual=model(x,torch.zeros(2,65,device="cuda",dtype=torch.float64))
        expected=native(x)
    # Native combined.py quantizes phases to BF16; this layer accumulates FP32.
    assert relative_error(actual,expected)<.01
