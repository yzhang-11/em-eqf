"""New mixer task configurations must work end to end with precise clocks."""
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from src.data.dot_dict import DotDict
from src.models.builders import ModelBuilder

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="Mamba3 GPU kernels required")


@pytest.mark.parametrize("task",["reg_mixer_attnpl_t","clf_mixer_attnpl_t"])
@pytest.mark.parametrize("suffix",["mamba3_rotary","mamba3_rotary_mimo"])
def test_full_mixer_task_config_forward_backward(task,suffix):
    root=Path(__file__).resolve().parents[2]
    cfg=OmegaConf.to_container(OmegaConf.load(root/f"config/{task}_{suffix}.yaml"),resolve=True)
    cfg["stats"]=dict(tau_mean=1.,tau_min=.01,tau_max=10.,tau_q025=.025,tau_q05=.05,
        tau_unfiltered=1.,mag_mean=3.5,b_value=1.,mag_completeness=3.)
    args=DotDict(cfg)
    device=torch.device("cuda:0")
    torch.manual_seed(101)
    model=ModelBuilder.by_name(task)()(args,device)
    observed=[]
    def clock_dtype(module,inputs,kwargs):
        observed.append(kwargs["times"].dtype)
    handle=model.base_model.encoder.register_forward_pre_hook(clock_dtype,with_kwargs=True)
    x=torch.zeros(2,65,6,device=device,dtype=torch.float64)
    gaps=torch.tensor([.3,1.1,.02,.4,.6]*13,device=device,dtype=torch.float64)
    clock=gaps.cumsum(0)
    x[:,:,0]=clock
    x[:,:,1]=clock/clock[-1]
    x[:,:,2]=3.+torch.rand(2,65,device=device,dtype=torch.float64)
    x[:,:,3:5]=1.
    x[:,:,-1]=gaps
    y=model(x)
    handle.remove()
    assert observed==[torch.float64]
    assert y.shape==(2,) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    mixer=model.base_model.encoder.layers[0].mixer
    assert mixer.in_proj.weight.grad[-mixer.num_rope_angles:].abs().sum()>0
    assert args.resume_path is None and args.load_specific_parts==[]
