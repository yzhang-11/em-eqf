import torch
from src.models.tpp.magnitude_state import MagnitudeStateModel, markov_predictive_weights


def test_parallel_filter_equals_sequential_and_gradients():
    torch.manual_seed(7)
    emission=torch.randn(73,3,dtype=torch.float64,requires_grad=True)
    transition=torch.randn(3,3,dtype=torch.float64).softmax(-1)
    initial=torch.ones(3,dtype=torch.float64)/3
    actual=markov_predictive_weights(emission,transition,initial)
    state=initial;values=[]
    for row in emission:
        values.append(state)
        posterior=(state.log()+row).softmax(-1)
        state=posterior @ transition
    expected=torch.stack(values)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    actual.square().sum().backward()
    assert torch.isfinite(emission.grad).all()


def test_current_and_future_marks_do_not_change_own_predictive_distribution():
    model=MagnitudeStateModel(2,3.,.1)
    marks=torch.tensor([3.,3.2,3.4,4.,3.1],dtype=torch.float64)
    previous,_=model.predictive_weights(marks)
    changed=marks.clone();changed[2:]=torch.tensor([6.,5.,4.])
    after,_=model.predictive_weights(changed)
    torch.testing.assert_close(previous[:3],after[:3],atol=0,rtol=0)
    assert not torch.allclose(previous[3],after[3])


def test_distribution_and_sequence_likelihood_agree():
    model=MagnitudeStateModel(2,2.5,.01)
    marks=torch.tensor([2.5,2.65,3.1,5.,2.9],dtype=torch.float64)
    logp=model.log_prob_sequence(marks)
    torch.testing.assert_close(logp,model.distribution_sequence(marks).log_prob(marks))
    (-logp.mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
