"""Independent numerical checks of the accelerated, unchanged ETAS likelihood."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/tuning')]
from fit_region_etas import FastETAS,RATES,CLASSIC_RATES,coefficients,direct_points,_history_states


def fixture():
    rng=np.random.default_rng(817)
    ns=np.rint(np.r_[0.,rng.lognormal(-1.5,1.5,479)].cumsum()*86400e9).astype(np.int64)
    mag=2.5+.1*rng.geometric(.23,480)
    return dict(ns=ns,magnitude=mag,metadata=[2.5,1.,1.,.01,10.,.1,0],train_end=384,
        val_end=432,test_end=480,block_starts=[256],block_targets=[128])


@pytest.mark.parametrize('mode',['full','window'])
@pytest.mark.parametrize('classic',[False,True])
def test_objective_gradient_and_direct_likelihood(mode,classic):
    model=FastETAS(fixture(),mode,classic)
    theta=np.array([-.3,np.log(.005),np.log(.3),1.2,1.2])
    if classic:theta[2]=np.log(.8);theta[3]=np.log(.02)
    value,gradient=model.objective(theta)
    finite=[]
    for j in range(5):
        a=theta.copy();b=theta.copy();a[j]+=1e-5;b[j]-=1e-5
        finite.append((model.objective(a)[0]-model.objective(b)[0])/2e-5)
    np.testing.assert_allclose(gradient,finite,rtol=2e-5,atol=1e-7)
    indices=np.arange(256,384)
    expected=direct_points(model.spec,model.describe(theta),indices,None if mode=='full' else 1024)
    np.testing.assert_allclose(value,expected.mean(),rtol=0,atol=2e-6)
    validation,points,_=model.evaluate(theta,'val',None if mode=='full' else 1024,True)
    expected=direct_points(model.spec,model.describe(theta),np.arange(384,432),None if mode=='full' else 1024)
    np.testing.assert_allclose(points,expected,rtol=1e-5,atol=2e-6)
    assert np.isfinite(validation['total'])


@pytest.mark.parametrize('p',[1.001,1.2,2.,5.])
@pytest.mark.parametrize('c',[1e-7,.001,.1,10.])
def test_quadrature_is_omori_kernel(p,c):
    theta=np.array([0.,np.log(c),np.log(p-1),0.,1.])
    *_,a,dc,dp,e,dn=coefficients(theta,1.,1.,2.5)
    t=np.logspace(-11,5,240)
    actual=np.exp(-t[:,None]*RATES)@a
    expected=(p-1)*c**(p-1)*(t+c)**(-p)
    assert np.max(np.abs(actual/expected-1))<5e-6


def test_exponential_history_is_causal():
    spec=fixture();dt=np.r_[0.,np.diff(spec['ns'])/86400e9]
    decay=np.exp(-dt[:,None]*RATES);weight=np.exp(1.2*(spec['magnitude']-2.5))
    all_states=_history_states(decay,weight)
    prefix=_history_states(decay[:300],weight[:300])
    np.testing.assert_array_equal(all_states[:300],prefix)


@pytest.mark.parametrize('p',[.2,.5,.99,1.,1.2,5.])
@pytest.mark.parametrize('c',[1e-7,.001,10.])
def test_classic_quadrature_includes_p_below_one(p,c):
    theta=np.array([0.,np.log(c),np.log(p),np.log(.02),1.])
    *_,a,dc,dp,e,dn=coefficients(theta,1.,1.,2.5,True)
    t=np.logspace(-11,5,240)
    actual=np.exp(-t[:,None]*CLASSIC_RATES)@a
    expected=(t+c)**(-p)
    assert np.max(np.abs(actual/expected-1))<5e-6


def test_agrees_with_existing_zhuang_etas():
    from src.data import Batch,Sequence
    from src.models.tpp.etas_zhuang import ETASZhuang
    model=FastETAS(fixture(),'full');theta=np.array([-.3,np.log(.005),np.log(.3),1.2,1.2])
    p=model.describe(theta);spec=model.spec;end=spec['train_end']
    arrival=(spec['ns'][:end]-spec['ns'][0])/86400e9+1.
    seq=Sequence(torch.from_numpy(np.diff(np.r_[0.,arrival,arrival[-1]])),
        t_nll_start=float(arrival[255]),mag=torch.from_numpy(spec['magnitude'][:end].copy()))
    seq.t_nll_start=float(seq.arrival_times[255])
    reference=ETASZhuang(omori_p_init=p['p'],omori_c_init=p['c'],base_rate_init=p['mu'],
        productivity_K_init=p['K'],productivity_alpha_e_init=p['alpha_e'],richter_b=p['b'],
        mag_completeness=p['mc'],report_params=False,device='cpu').double()
    with torch.no_grad():
        for name,value in [('log_p_minus_one',np.log(p['p']-1)),('log_c',np.log(p['c'])),
            ('log_mu',np.log(p['mu'])),('log_K',np.log(p['K'])),('alpha_e_param',p['alpha_e'])]:
            getattr(reference,name).fill_(value)
    expected=float(reference.nll_loss(Batch.from_list([seq]),reduction='per_event'))
    np.testing.assert_allclose(model.objective(theta)[0],expected,rtol=0,atol=2e-6)


@pytest.mark.parametrize('p',[.8,1.3])
def test_agrees_with_existing_classic_etas(p):
    from src.data import Batch,Sequence
    from src.models.tpp.etas import ETAS
    model=FastETAS(fixture(),'full',True)
    theta=np.array([-.3,np.log(.005),np.log(p),np.log(.02),1.2]);params=model.describe(theta)
    spec=model.spec;end=spec['train_end'];arrival=(spec['ns'][:end]-spec['ns'][0])/86400e9+1.
    seq=Sequence(torch.from_numpy(np.diff(np.r_[0.,arrival,arrival[-1]])),
        mag=torch.from_numpy(spec['magnitude'][:end].copy()))
    seq.t_nll_start=float(seq.arrival_times[255])
    reference=ETAS(omori_p_init=p,omori_c_init=params['c'],base_rate_init=params['mu'],
        productivity_k_init=params['K'],productivity_alpha_init=params['alpha_e']/np.log(10),
        richter_b=params['b'],mag_completeness=params['mc'],report_params=False,device='cpu').double()
    with torch.no_grad():
        for name,value in [('log_p',np.log(p)),('log_c',np.log(params['c'])),('log_mu',np.log(params['mu'])),
            ('log_k',np.log(params['K'])),('log_alpha',np.log(params['alpha_e']/np.log(10)))]:
            getattr(reference,name).fill_(value)
    expected=float(reference.nll_loss(Batch.from_list([seq]),reduction='per_event'))
    np.testing.assert_allclose(model.objective(theta)[0],expected,rtol=0,atol=2e-6)
