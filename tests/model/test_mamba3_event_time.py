"""The renamed default profile must reproduce the measured SCEDC mechanism."""
import importlib
import json
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("mamba_ssm.modules.mamba3")
from src.models.mamba.mamba3_event_time import Mamba3EventTime
from src.models.mamba.mamba3_rotary import Mamba3Rotary
from src.models.mamba.mixer_seq import MixerModel

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="native GPU kernels")
STORE=Path("/root/autodl-fs/em_eqf_training/scedc_full_time_dynamics_20261105")

def test_legacy_defaults_and_backend_alias():
    old=Mamba3Rotary(d_model=64, d_state=64, headdim=32)
    new=Mamba3EventTime(d_model=64)
    assert old.time_update_mode=="base" and old.use_event_time_phase and not old.is_mimo
    assert new.time_update_mode=="decay_only" and not new.use_event_time_phase
    assert new.is_mimo and new.mimo_rank==2
    assert importlib.import_module("src.models.mamba.mamba3_rotary_mimo") is importlib.import_module("src.models.mamba.mamba3_mimo_kernels")

@CUDA
def test_default_profile_matches_explicit_legacy_and_decodes():
    from mamba_ssm.utils.generation import InferenceParams
    torch.manual_seed(123)
    new=Mamba3EventTime(d_model=64,layer_idx=0,device="cuda").eval()
    old=Mamba3Rotary(d_model=64,d_state=64,headdim=32,expand=2,rope_fraction=.5,
        is_mimo=True,mimo_rank=2,chunk_size=16,time_update_mode="decay_only",layer_idx=0,device="cuda").eval()
    old.load_state_dict(new.state_dict(),strict=True)
    x=torch.randn(1,65,64,device="cuda",requires_grad=True)
    g=torch.linspace(.01,4,65,device="cuda",dtype=torch.float64)[None]
    t=g.cumsum(1)
    expected=old(x,t,physical_dt=g,phase_times=torch.zeros_like(t))
    actual=new(x,t,physical_dt=g)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    actual.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
    cache=InferenceParams(max_seqlen=80,max_batch_size=1)
    with torch.no_grad():
        a=new(x[:,:59],t[:,:59],physical_dt=g[:,:59],inference_params=cache)
        cache.seqlen_offset=59
        b=new(x[:,59:],t[:,59:],physical_dt=g[:,59:],inference_params=cache)
        error=(torch.cat([a,b],1)-actual).norm()/actual.norm()
        assert error<.035

@CUDA
def test_tpp_builder_adapter_and_saved_checkpoint():
    from src.models.builders import ModelBuilder
    from src.data import Batch,Sequence
    payload=torch.load(STORE/"dataset_snapshot.pt",map_location="cpu",weights_only=False)
    cfg=OmegaConf.load("config/mixer_tpp_mamba3_event_time.yaml")
    cfg.update(payload["stats"])
    new=ModelBuilder.by_name("mixer_tpp")()(cfg,torch.device("cuda:0"))
    original=torch.load(STORE/"runs/decay_only_native_seed_0/best_model.pth",map_location="cuda",weights_only=False)
    new.load_state_dict(original["model_state_dict"],strict=True)
    state=payload["sequences"]["train"]
    seq=Sequence(inter_times=state["inter_times"][:66],mag=state["mag"][:65],t_start=0.)
    batch=Batch.from_list([seq]).to(torch.device("cuda:0"))
    adapted=new.base_model.input_adapter(batch)
    names=new.base_model.input_adapter.features_input_keys
    from src.models.adapters.mixer_tpp import MixerBatchAdapter
    legacy_cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    legacy_cfg.mixer_model_config.ssm_cfg.layer="Mamba3Rotary"
    legacy_cfg.mask_input_features=[]
    original_features=MixerBatchAdapter(legacy_cfg)(batch)["features"]
    assert torch.count_nonzero(adapted["features"][...,names.index("log_inter_times")])==0
    assert torch.equal(adapted["features"][...,names.index("mag")],original_features[...,names.index("mag")])
    assert adapted["times"].dtype==torch.float64
    assert (adapted["physical_dt"]>0).all()
    assert adapted["physical_dt"][0,-1]==1
    assert sum(p.numel() for p in new.parameters())==154097
    loss=new.nll_loss(batch,reduction="per_event")["total"].mean()
    loss.backward()
    assert torch.isfinite(loss)
    # The mixer constructor must dispatch the new public name.
    assert all(isinstance(layer.mixer,Mamba3EventTime) for layer in new.base_model.encoder.layers)


@pytest.mark.parametrize("name", ["Mamba3Rotary", "Mamba3EventTime"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("physical", [False, True])
def test_input_controls_are_independent_of_model_name(name,masked,physical):
    from src.models.adapters.mixer_tpp import MixerBatchAdapter
    from src.data import Batch,Sequence
    cfg=OmegaConf.load("config/mixer_tpp_mamba3_event_time.yaml")
    cfg.update(dict(tau_mean=.5,tau_min=.01,tau_max=3.,tau_q05=.1,tau_q025=.02,
        mag_mean=2.5,time_max=10.,richter_b_mle=1.,mag_completeness=2.))
    cfg.mixer_model_config.ssm_cfg.layer=name
    cfg.mask_input_features=["log_inter_times"] if masked else []
    cfg.provide_physical_dt=physical
    seq=Sequence(inter_times=torch.tensor([.1,.6,.2,.5],dtype=torch.float64),
                 mag=torch.tensor([2.1,2.4,2.3]),t_start=0.)
    batch=Batch.from_list([seq])
    result=MixerBatchAdapter(cfg)(batch)
    cfg.mask_input_features=[]
    baseline=MixerBatchAdapter(cfg)(batch)
    assert torch.equal(result["times"],baseline["times"])
    assert torch.equal(result["features"][...,1],baseline["features"][...,1])
    if masked:assert torch.count_nonzero(result["features"][...,0])==0
    else:assert torch.equal(result["features"],baseline["features"])
    assert ("physical_dt" in result)==physical
    if physical:
        assert torch.equal(result["physical_dt"][0,:3],batch.inter_times[0,:3]/.5)
        assert result["physical_dt"][0,-1]==1
