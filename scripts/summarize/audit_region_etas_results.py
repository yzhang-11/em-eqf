"""Independent exact-kernel checks on final held-out ETAS forecasts."""
from pathlib import Path
import json
import math
import sys

import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT/'scripts/tuning'))
import fit_region_etas as etas


def direct(spec,parameters,indices,context):
    mu=parameters['mu'];K=parameters['K'];p=parameters['p'];c=parameters['c']
    alpha=parameters['alpha_e'];mc=parameters['mc'];origin=spec['val_end']
    norm=1. if parameters.get('kernel')=='classic' else (p-1)*c**(p-1)
    result=[]
    for index in indices:
        first=origin+((index-origin)//128)*128
        left=0 if context is None else max(1,first-context)
        past=np.arange(left,index)
        now=(spec['ns'][index]-spec['ns'][past])/86400e9
        previous=(spec['ns'][index-1]-spec['ns'][past])/86400e9
        productivity=np.exp(alpha*(spec['magnitude'][past]-mc))
        intensity=mu+K*np.sum(productivity*norm*(now+c)**(-p))
        if abs(p-1)<1e-8:
            mass=np.log1p((now-previous)/(previous+c))
        else:
            log_ratio=np.log1p((now-previous)/(previous+c))
            mass=(previous+c)**(1-p)*(-np.expm1((1-p)*log_ratio))/(p-1)
        integral=mu*((spec['ns'][index]-spec['ns'][index-1])/86400e9)
        integral+=K*norm*np.sum(productivity*mass)
        result.append(integral-math.log(intensity))
    return np.asarray(result)


def main():
    torch.set_num_threads(1)
    assert (etas.OUT/'test_results.json').exists()
    data=torch.load(etas.DATA,map_location='cpu',weights_only=False)
    rows=[]
    for region,spec in sorted(data['regions'].items()):
        for mode in ['full','window']:
            candidates={classic:json.loads((etas.OUT/f"{region}_{mode}{'_classic' if classic else ''}.json").read_text())
                        for classic in [False,True]}
            classic=min(candidates,key=lambda k:candidates[k]['validation']['total'])
            fit=candidates[classic];model=etas.FastETAS(spec,mode,classic)
            context=None if mode=='full' else 1024
            _,points,_=model.evaluate(np.asarray(fit['selected']['theta']),'test',context,return_points=True)
            indices=np.unique(np.linspace(spec['val_end'],spec['test_end']-1,32).astype(int))
            exact=direct(spec,fit['parameters'],indices,context)
            error=float(np.max(np.abs(points[indices-spec['val_end']]-exact)))
            assert error<1e-4,(region,mode,error)
            rows.append(dict(region=region,mode=mode,family=fit.get('family','subcritical'),
                queries=len(indices),max_absolute_time_nll_error=error))
            del model
    result=dict(passed=True,queries=sum(v['queries'] for v in rows),
        maximum_absolute_time_nll_error=max(v['max_absolute_time_nll_error'] for v in rows),rows=rows)
    (etas.OUT/'direct_test_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
