"""Full-history and history-matched classic/subcritical ETAS via Gamma quadrature.

The five-parameter Omori models are unchanged. A fixed log-rate
quadrature and recursive exponential states avoid quadratic event pairs.
Reference: Ogata, Matsu'ura & Katsura (1993), doi:10.1029/93GL02142.
"""
import argparse
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import digamma,expit,gammaln,logsumexp

ROOT=Path('/root/autodl-tmp/em_eqf')
REPORTS=ROOT/'experiments/revision/multiregion_pretraining/reports/region_capacity_20261105'
OUT=REPORTS/'etas'
DATA=ROOT/'data/MultiRegion/catalogs/region_capacity_20261105.pt'
RATES=np.exp(np.linspace(-14*math.log(10),9*math.log(10),160))
CLASSIC_RATES=np.exp(np.linspace(-60*math.log(10),9*math.log(10),448))
LOG_STEP=np.log(RATES[1]/RATES[0])


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def mle_b(magnitude,mc):
    mean=float(np.mean(np.rint((magnitude-mc)/.1)));bins=round((10-mc)/.1)
    low,high=.2,2.
    for _ in range(80):
        b=(low+high)/2;rate=b*np.log(10)*.1
        expected=1/np.expm1(rate)-bins/np.expm1(bins*rate)
        if expected>mean:low=b
        else:high=b
    return (low+high)/2


def mark_nll(magnitude,mc,b):
    rate=b*np.log(10)*.1;bins=round((10-mc)/.1)
    return -(np.log(-np.expm1(-rate))-np.log(-np.expm1(-bins*rate))-rate*np.rint((magnitude-mc)/.1))


def coefficients(theta,base_rate,b,mc,classic=False):
    mu=base_rate*np.exp(theta[0]);c=np.exp(theta[1]);p=np.exp(theta[2]) if classic else 1+np.exp(theta[2]);fraction=expit(theta[3])
    n=.995*fraction;alpha=theta[4];rates=CLASSIC_RATES if classic else RATES
    k=np.arange(round((10-mc)/.1),dtype=float);excess=.1*k;beta=b*np.log(10)
    logits=-beta*excess+alpha*excess
    log_e=logsumexp(logits)-logsumexp(-beta*excess)
    expected_excess=float(np.sum(np.exp(logits-logsumexp(logits))*excess))
    weights=np.ones(len(rates))*np.log(rates[1]/rates[0]);weights[[0,-1]]*=.5
    if classic:
        K=np.exp(theta[3])
        n=float(np.exp(theta[3]+(1-p)*np.log(c)-np.log(p-1)+log_e)) if p>1 else None
        a=np.exp(-gammaln(p)+p*np.log(rates)-c*rates)*weights
        dc=-a*c*rates;dp=a*p*(np.log(rates)-digamma(p))
        return mu,c,p,n,alpha,K,a,dc,dp,0.,1.
    K=n*np.exp(-log_e)
    a=np.exp(np.log(p-1)+(p-1)*np.log(c)-gammaln(p)+p*np.log(rates)-c*rates)*weights
    dc=a*((p-1)-c*rates)
    dp=a*(1+(p-1)*(np.log(c)-digamma(p)+np.log(rates)))
    return mu,c,p,n,alpha,K,a,dc,dp,expected_excess,1-fraction


@njit(cache=True)
def _loss_gradient(decay,phi,dt,x,weight,score,reset,a,dc,dp,mu,K,e_alpha,dn):
    states=np.zeros(len(a));derivative=np.zeros(len(a));gradient=np.zeros(5)
    loss=0.;count=0
    for i in range(len(dt)):
        if reset[i]:
            states[:]=0.;derivative[:]=0.;previous=0.;previous_x=0.
        else:previous=weight[i-1];previous_x=x[i-1]
        L=0.;I=0.;Lc=0.;Ic=0.;Lp=0.;Ip=0.;La=0.;Ia=0.
        for q in range(len(a)):
            post=states[q]+previous;dpost=derivative[q]+previous*previous_x
            current=decay[i,q]*post;dcurrent=decay[i,q]*dpost
            if score[i]:
                integral=phi[i,q]*post;dintegral=phi[i,q]*dpost
                L+=a[q]*current;I+=a[q]*integral
                Lc+=dc[q]*current;Ic+=dc[q]*integral
                Lp+=dp[q]*current;Ip+=dp[q]*integral
                La+=a[q]*dcurrent;Ia+=a[q]*dintegral
            states[q]=current;derivative[q]=dcurrent
        if score[i]:
            intensity=mu+K*L
            loss+=mu*dt[i]+K*I-math.log(intensity);count+=1
            gradient[0]+=mu*dt[i]-mu/intensity
            gradient[1]+=K*Ic-K*Lc/intensity
            gradient[2]+=K*Ip-K*Lp/intensity
            gradient[3]+=dn*(K*I-K*L/intensity)
            gradient[4]+=K*(Ia-e_alpha*I)-K*(La-e_alpha*L)/intensity
    return loss/count,gradient/count


@njit(cache=True)
def _history_states(decay,weight):
    states=np.zeros(decay.shape[1]);out=np.empty_like(decay)
    for i in range(len(weight)):
        previous=0. if i==0 else weight[i-1]
        for q in range(len(states)):
            states[q]=decay[i,q]*(states[q]+previous);out[i,q]=states[q]
    return out


class FastETAS:
    def __init__(self,spec,mode='full',classic=False):
        self.spec=spec;self.mode=mode;self.mc=spec['metadata'][0];self.classic=classic
        self.rates=CLASSIC_RATES if classic else RATES
        end=spec['train_end'];self.b=mle_b(spec['magnitude'][256:end],self.mc)
        self.base_rate=(end-256)/((spec['ns'][end-1]-spec['ns'][255])/86400e9)
        if mode=='full':
            ids=np.arange(end);reset=np.zeros(end,dtype=np.bool_);reset[0]=True;score=ids>=256
            dt=np.r_[0.,np.diff(spec['ns'][:end])/86400e9]
        else:
            ids=[];dt=[];reset=[];score=[]
            for first,targets in zip(spec['block_starts'],spec['block_targets']):
                left=max(1,first-1024);right=first+targets;index=np.arange(left,right)
                ids.extend(index);dt.extend(np.r_[0.,np.diff(spec['ns'][left:right])/86400e9])
                reset.extend([True]+[False]*(len(index)-1));score.extend(index>=first)
            ids=np.asarray(ids);dt=np.asarray(dt);reset=np.asarray(reset);score=np.asarray(score)
        self.dt=np.asarray(dt,dtype=np.float64);self.x=np.ascontiguousarray(spec['magnitude'][ids]-self.mc)
        self.reset=np.asarray(reset,dtype=np.bool_);self.score=np.asarray(score,dtype=np.bool_)
        self.decay=np.exp(-self.dt[:,None]*self.rates)
        self.phi=-np.expm1(-self.dt[:,None]*self.rates)/self.rates

    def objective(self,theta):
        mu,c,p,n,alpha,K,a,dc,dp,e_alpha,dn=coefficients(theta,self.base_rate,self.b,self.mc,self.classic)
        return _loss_gradient(self.decay,self.phi,self.dt,self.x,np.exp(alpha*self.x),
            self.score,self.reset,a,dc,dp,mu,K,e_alpha,dn)

    def describe(self,theta):
        mu,c,p,n,alpha,K,*_=coefficients(theta,self.base_rate,self.b,self.mc,self.classic)
        return dict(mu=mu,c=c,p=p,branching_ratio=n,alpha_e=alpha,K=K,b=self.b,mc=self.mc,
                    kernel='classic' if self.classic else 'normalized_subcritical')

    def evaluate(self,theta,split,context=None,return_points=False):
        spec=self.spec
        mu,c,p,n,alpha,K,a,*_=coefficients(theta,self.base_rate,self.b,self.mc,self.classic)
        # Reading a future suffix does not enter any prefix state.
        high=spec['val_end'] if split=='val' else spec['test_end']
        low=spec['train_end'] if split=='val' else spec['val_end']
        ns=spec['ns'][:high];dt=np.r_[0.,np.diff(ns)/86400e9]
        decay=np.exp(-dt[:,None]*self.rates);phi=-np.expm1(-dt[:,None]*self.rates)/self.rates
        weight=np.exp(alpha*(spec['magnitude'][:high]-self.mc));states=_history_states(decay,weight)
        prior=np.vstack([np.zeros((1,len(self.rates))),states[:-1]+weight[:-1,None]])
        time_loss=np.empty(high-low)
        for first in range(low,high,128):
            right=min(first+128,high);indices=np.arange(first,right)
            current=states[indices].copy();integrand=prior[indices].copy()
            if context is not None:
                left=max(1,first-context)
                current-=np.exp(-((ns[indices]-ns[left])/86400e9)[:,None]*self.rates)*states[left]
                integrand-=np.exp(-((ns[indices-1]-ns[left])/86400e9)[:,None]*self.rates)*states[left]
            intensity=mu+K*(current@a)
            integral=mu*dt[indices]+K*np.sum(phi[indices]*integrand*a,axis=1)
            assert np.all(intensity>0) and np.all(np.isfinite(intensity))
            time_loss[first-low:right-low]=integral-np.log(intensity)
        mag_loss=mark_nll(spec['magnitude'][low:high],self.mc,self.b)
        result=dict(time=float(time_loss.mean()),mag=float(mag_loss.mean()),
                    total=float((time_loss+mag_loss).mean()),events=high-low)
        return (result,time_loss,states) if return_points else result


def direct_points(spec,parameters,indices,context=None):
    """Independent O(N^2) analytic reference, used only for a few audit queries."""
    mu=parameters['mu'];K=parameters['K'];p=parameters['p'];c=parameters['c'];alpha=parameters['alpha_e'];mc=parameters['mc']
    result=[];low=spec['train_end']
    for index in indices:
        first=low+((index-low)//128)*128
        left=0 if context is None else max(1,first-context)
        past=np.arange(left,index)
        now=(spec['ns'][index]-spec['ns'][past])/86400e9
        previous=(spec['ns'][index-1]-spec['ns'][past])/86400e9
        weight=np.exp(alpha*(spec['magnitude'][past]-mc))
        classic=parameters.get('kernel')=='classic'
        norm=1. if classic else (p-1)*c**(p-1)
        intensity=mu+K*np.sum(weight*norm*(now+c)**(-p))
        log_ratio=np.log((now+c)/(previous+c))
        mass=(log_ratio if abs(p-1)<1e-8 else
              np.exp((1-p)*np.log(previous+c))*(-np.expm1((1-p)*log_ratio))/(p-1))
        mass*=norm
        integral=mu*((spec['ns'][index]-spec['ns'][index-1])/86400e9)+K*np.sum(weight*mass)
        result.append(integral-np.log(intensity))
    return np.asarray(result)


def fit_one(region,spec,mode,classic):
    suffix='_classic' if classic else ''
    path=OUT/f'{region}_{mode}{suffix}.json'
    if path.exists():return
    model=FastETAS(spec,mode,classic);started=time.time();results=[]
    bounds=([(-12,8),(math.log(1e-7),math.log(10)),(math.log(.2),math.log(5)),(-90,30),(0,5)] if classic else
            [(-12,8),(math.log(1e-7),math.log(10)),(math.log(.001),math.log(4)),(-8,8),(0,5)])
    for restart,(p,c,n,relative_alpha) in enumerate([(1.1,.001,.5,.7),(1.5,.03,.8,.4)]):
        theta=np.array([math.log(.4),math.log(c),math.log(p-1),math.log(n/(.995-n)),relative_alpha*model.b*math.log(10)])
        if classic:
            K=coefficients(theta,model.base_rate,model.b,model.mc)[5]
            theta[2]=math.log(p);theta[3]=math.log(K*(p-1)*c**(p-1))
        initial=float(model.objective(theta)[0])
        result=minimize(model.objective,theta,method='L-BFGS-B',jac=True,bounds=bounds,
            options=dict(maxiter=300,ftol=1e-10,gtol=1e-5,maxls=30))
        results.append(dict(theta=result.x.tolist(),train_time_nll=float(result.fun),initial_time_nll=initial,
            success=bool(result.success),message=str(result.message),iterations=int(result.nit),function_calls=int(result.nfev),
            gradient_norm=float(np.linalg.norm(result.jac))))
        print('ETAS_RESTART',region,mode,suffix,restart,results[-1],flush=True)
    selected=min(results,key=lambda v:v['train_time_nll']);theta=np.array(selected['theta'])
    assert np.all(np.isfinite(theta)) and np.isfinite(selected['train_time_nll'])
    context=None if mode=='full' else 1024
    validation,points,_=model.evaluate(theta,'val',context,return_points=True)
    queries=np.unique(np.linspace(spec['train_end'],spec['val_end']-1,min(32,spec['val_end']-spec['train_end'])).astype(int))
    exact=direct_points(spec,model.describe(theta),queries,context)
    error=float(np.max(np.abs(points[queries-spec['train_end']]-exact)))
    assert error<1e-4,(region,mode,error)
    save(path,dict(region=region,mode=mode,family='classic' if classic else 'subcritical',selected=selected,restarts=results,
        parameters=model.describe(theta),validation=validation,base_rate=model.base_rate,
        exact_validation_queries=len(queries),max_absolute_nll_quadrature_error=error,
        quadrature_nodes=len(model.rates),seconds=time.time()-started,
        training_events=spec['train_end']-256,test_accessed=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('ETAS_COMPLETE',region,mode,suffix,validation,'error',error,flush=True)


def fit_all():
    import torch
    torch.set_num_threads(1)
    data=torch.load(DATA,map_location='cpu',weights_only=False);OUT.mkdir(parents=True,exist_ok=True)
    for region,spec in sorted(data['regions'].items()):
        for mode in ['full','window']:
            for classic in [False,True]:fit_one(region,spec,mode,classic)
    save(OUT/'fit_complete.json',dict(completed_regions=len(data['regions']),fits=4*len(data['regions']),
        test_accessed=False,method='Classic unconstrained and normalized subcritical ETAS; full and matched histories; Gamma quadrature.',
        selection='Per-history family selected with validation only; all optima fitted to training likelihood.',
        reference='https://doi.org/10.1029/93GL02142'))


def finalize():
    import torch
    data=torch.load(DATA,map_location='cpu',weights_only=False);assert (OUT/'fit_complete.json').exists()
    rows=[]
    for region,spec in sorted(data['regions'].items()):
        for mode in ['full','window']:
            candidates={classic:json.loads((OUT/f"{region}_{mode}{'_classic' if classic else ''}.json").read_text()) for classic in [False,True]}
            selected=min(candidates,key=lambda classic:candidates[classic]['validation']['total'])
            for classic,result in candidates.items():
                model=FastETAS(spec,mode,classic)
                contexts=[None,64,1024,4096] if mode=='full' else [64,1024,4096]
                for context in contexts:
                    metrics=model.evaluate(np.array(result['selected']['theta']),'test',context)
                    rows.append(dict(region=region,mode=mode,family='classic' if classic else 'subcritical',
                        validation_selected=classic==selected,eval_context='full' if context is None else context,**metrics))
    save(OUT/'test_results.json',dict(rows=rows,units='nats/event',lower_is_better=True))
    print('ETAS_TESTS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',action='store_true');parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.fit:fit_all()
    elif args.finalize:finalize()
    else:parser.error('Choose --fit or --finalize')
