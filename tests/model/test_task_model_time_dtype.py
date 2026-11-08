"""Preserve precise encoder timestamps while matching the task pool's dtype."""
import pytest
import torch
from torch import nn
from src.models.core.task_model import TaskModel
from src.models.extractors.attn_time_biased_mh import TimeAwareAttnPoolMH


class Adapter:
    def get_extra_inputs(self,x):
        return {"event_time":x[:,:,1],"ids":torch.arange(x.shape[1],device=x.device)}


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_adapter=Adapter()
        self.projection=nn.Linear(1,4)
        self.received_dtype=None
    def forward(self,x,caches=None):
        self.received_dtype=x.dtype
        hidden=self.projection(x[:,:,:1].float())
        mask=torch.ones(x.shape[:2],device=x.device,dtype=torch.bool)
        return hidden,mask,None


class Pool(TimeAwareAttnPoolMH):
    def forward(self,x,mask,extra_inputs=None,**kwargs):
        assert extra_inputs["event_time"].dtype==x.dtype
        assert extra_inputs["ids"].dtype==torch.int64
        return super().forward(x,mask,extra_inputs,**kwargs)


@pytest.mark.parametrize("method",["forward","get_pooled_representation"])
def test_double_timestamps_are_only_cast_at_task_pooling(method):
    torch.manual_seed(103)
    base=Base()
    pool=Pool(d_model=4,d_hidden=4,n_heads=1,agg="mean")
    model=TaskModel(base,pool,nn.Linear(pool.output_dim,1))
    x=torch.rand(2,8,2,dtype=torch.float64,requires_grad=True)
    output=getattr(model,method)(x)
    assert output.dtype==torch.float32 and torch.isfinite(output).all()
    assert base.received_dtype==torch.float64 and x.dtype==torch.float64
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
