import math
import numpy as np
import pytest
import torch

from src.models.tpp.magnitude_density import MagnitudeGridDistribution,inverse_positive
from src.models.tpp.magnitude_dynamics import (
    contraction_bounds,filter_gas,gr_statistics,mixture_statistics,parameters,
)


@pytest.mark.parametrize('step,width',[(.1,7.),(.01,7.5)])
def test_gr_probability_and_score_match_normalized_distribution(step,width):
    state=.4
    bins=np.arange(round(width/step))*step
    result=np.array([gr_statistics(state,x,step,width) for x in bins])
    assert abs(np.exp(result[:,0]).sum()-1.)<1e-12
    b=result[0,3]
    dist=MagnitudeGridDistribution(torch.tensor([[inverse_positive(b*math.log(10.))]],dtype=torch.float64),
        'gr',1,10.-width,step)
    np.testing.assert_allclose(dist.log_prob(torch.tensor(10.-width+bins)).numpy(),result[:,0],atol=1e-12)
    epsilon=1e-5
    numerical=np.array([(gr_statistics(state+epsilon,x,step,width)[0]-gr_statistics(state-epsilon,x,step,width)[0])/(2*epsilon) for x in bins])
    np.testing.assert_allclose(numerical,result[:,1],atol=3e-9)
    assert abs(np.sum(np.exp(result[:,0])*result[:,1]))<1e-12


def test_mixture_score_and_curvature():
    state=.3;first=math.log(.2);second=math.log(.1);eps=1e-5
    lp,score,curvature,_=mixture_statistics(state,first,second)
    upper=mixture_statistics(state+eps,first,second)
    lower=mixture_statistics(state-eps,first,second)
    assert abs((upper[0]-lower[0])/(2*eps)-score)<1e-10
    assert abs((upper[1]-lower[1])/(2*eps)-curvature)<1e-10


@pytest.mark.parametrize('family',['gr_free','gr_contractive','mixture_contractive'])
def test_current_and_future_marks_do_not_affect_own_state(family):
    marks=np.array([0.,.2,1.,.1,2.])
    components=np.array([[-1.,-2.],[-2.,-1.],[-3.,-4.],[-1.,-2.],[-2.,-3.]])
    theta=[.3,math.log(32.),math.log(.1)]
    old=filter_gas(theta,family,marks,.1,7.,components=components,return_trace=True)
    newer=marks.copy();newer[2:]+=1.
    changed=components.copy();changed[2:]=changed[2:,::-1]
    new=filter_gas(theta,family,newer,.1,7.,components=changed,return_trace=True)
    np.testing.assert_array_equal(old['state'][:3],new['state'][:3])
    assert old['state'][3]!=new['state'][3]


@pytest.mark.parametrize('family,cap',[('gr_contractive',.3),('mixture_contractive',2.)])
def test_contraction_bound_and_initial_state_forgetting(family,cap):
    theta=[.3,math.log(2.),math.log(cap)]
    low,high=contraction_bounds(theta,family,7.5)
    bound=max(abs(low),abs(high));assert bound<1.
    marks=np.random.default_rng(71).uniform(0,7.5,30)
    components=np.random.default_rng(72).normal(-3.,2.,(30,2))
    one=filter_gas(theta,family,marks,.01,7.5,components=components,initial=-3.,return_trace=True)
    two=filter_gas(theta,family,marks,.01,7.5,components=components,initial=3.,return_trace=True)
    assert (one['update_jacobian']>=low).all() and (one['update_jacobian']<=high).all()
    delta=abs(one['state']-two['state'])
    assert np.all(delta[1:]<=bound*delta[:-1]+1e-12)
