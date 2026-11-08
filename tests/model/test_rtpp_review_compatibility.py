"""Baseline precision and sampler-state checks for the frozen catalog protocol."""
from types import SimpleNamespace

import pytest
import torch

from src.data import Batch,Sequence
from src.models.builders import ModelBuilder


def build(use_double=False):
    args=SimpleNamespace(d_model=8,num_components=2,num_rnn_layers=1,tau_mean=1.,mag_mean=3.2,
        time_max=10.,richter_b_mle=1.,mag_completeness=3.,rnn_type='GRU',rnn_dropout=0.,
        rnn_chunk_len=64,use_double_precision=use_double,loss_reduction='per_time')
    return ModelBuilder.by_name('rtpp')()(args,torch.device('cpu')).eval()


@pytest.mark.parametrize('use_double',[False,True])
def test_recast_builder_respects_precision_without_mutating_clock(use_double):
    model=build(use_double)
    dtype=torch.float64 if use_double else torch.float32
    assert next(model.parameters()).dtype==dtype
    seq=Sequence(inter_times=torch.tensor([1.,2.,.5],dtype=dtype),
        mag=torch.tensor([3.2,3.4],dtype=dtype))
    batch=Batch.from_list([seq]);clock=batch.inter_times.clone()
    loss=model.nll_loss(batch,return_dict=True,reduction='per_time')['total'].mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.testing.assert_close(batch.inter_times,clock,rtol=0,atol=0)
    assert batch.inter_times.dtype==dtype


def test_recast_sampling_hidden_matches_observed_history_only():
    torch.manual_seed(27)
    model=build()
    seq=Sequence(inter_times=torch.tensor([1.,2.,.5]),mag=torch.tensor([3.2,3.4]))
    batch=Batch.from_list([seq])
    with torch.no_grad():
        context,hidden=model._get_context_and_hidden(batch)
        features=torch.cat([model.encode_time(batch.inter_times[:,:-1]),
            model.encode_magnitude(batch.mag[:,:-1])],dim=-1)*batch.input_mask[:,:-1,None]
        expected,expected_hidden=model._run_rnn(features)
    torch.testing.assert_close(context[:,-1],expected[:,-1],rtol=0,atol=0)
    torch.testing.assert_close(hidden,expected_hidden,rtol=0,atol=0)


def test_recast_empty_history_has_zero_hidden_state():
    model=build()
    seq=Sequence(inter_times=torch.tensor([2.]),mag=torch.empty(0))
    context,hidden=model._get_context_and_hidden(Batch.from_list([seq]))
    assert context.shape==(1,1,model.context_size)
    assert hidden.shape==(1,1,model.context_size)
    assert torch.count_nonzero(context)==torch.count_nonzero(hidden)==0
