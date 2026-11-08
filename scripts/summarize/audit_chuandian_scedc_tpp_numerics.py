"""Audit the single tied test event without changing primary fits or scores."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

STORE=Path('/root/autodl-fs/em_eqf_training/chuandian_scedc_tpp_20261105')
sys.path.insert(0,str(STORE))
import frozen_summary as summary
study,c,REPORTS=summary.study,summary.c,summary.REPORTS


def read(path):return json.loads(Path(path).read_text())


def tied_spec():
    data=torch.load(study.DATA,map_location='cpu',weights_only=False)
    spec=data['regions']['scedc']
    ties=np.flatnonzero(np.diff(spec['physical_ns'])==0)+1
    assert len(ties)==1
    index=int(ties[0]);assert spec['val_end']<=index<spec['test_end']
    assert 0<(spec['ns'][index]-spec['ns'][index-1])/86400e9<1e-10
    return data,spec,index


def etas_points():
    _,spec,index=tied_spec();etas=summary.etas_module();rows=[]
    for result in read(REPORTS/'etas/test_results.json')['rows']:
        if result['target']!='scedc':continue
        mode=result['method'].removeprefix('etas_')
        suffix='_classic' if result['family']=='classic' else ''
        fit=read(REPORTS/f'etas/scedc_{mode}{suffix}.json')
        point=float(etas.direct_points(dict(spec,train_end=spec['val_end']),fit['parameters'],[index],1024 if mode=='window' else None)[0])
        mark=float(etas.mark_nll(spec['magnitude'][[index]],spec['metadata'][0],fit['parameters']['b'])[0])
        n=result['events']
        rows.append(dict(**result,tied_time_nll=point,tied_mark_nll=mark,
            without_tie_total=(result['total']*n-point-mark)/(n-1)))
    c.save(REPORTS/'etas/tied_event_audit.json',dict(rows=rows))


@torch.no_grad()
def main():
    assert read(REPORTS/'final_audit.json')['passed']
    plan=read(study.CONFIGS/'study_plan.json')
    subprocess.run([plan['etas_environment'],str(Path(__file__)),'--etas'],check=True)
    study.setup();data,spec,index=tied_spec()
    from run_multidomain_pretraining import make_batch
    first=spec['val_end']+((index-spec['val_end'])//128)*128
    batch=make_batch([c.sequence(spec,data['vocabulary']['scedc'],first,min(128,spec['test_end']-first),1024)])
    selected=(batch.event_index==index)&batch.nll_event_mask.bool()
    assert int(selected.sum())==1
    frame=pd.read_csv(REPORTS/'neural_metrics.csv');rows=[]
    for _,row in frame.loc[frame.target.eq('scedc')].iterrows():
        seed=int(row.seed);label=f'scedc_{row.family}_{row.method}_seed_{seed}'
        saved=read(REPORTS/f'neural/{label}.json')
        assert c.sha(saved['checkpoint'])==saved['checkpoint_sha256']
        model,_=study.build(row.family,seed,data['vocabulary'])
        model.load_state_dict(torch.load(saved['checkpoint'],map_location='cuda:0',weights_only=False)['model_state_dict'],strict=True)
        model.eval();context=model.get_context(batch)
        distribution=model.get_inter_time_dist(context)
        old=float((-distribution.log_prob(batch.inter_times.clamp_min(1e-10)))[selected].item())
        # Both the likelihood call and each Weibull component clamp time.
        assert distribution.component_distribution.eps==1e-10
        distribution.component_distribution.eps=1e-12
        lower=float((-distribution.log_prob(batch.inter_times.clamp_min(1e-12)))[selected].item())
        distribution.component_distribution.eps=1e-10
        mark=float((-model.get_magnitude_dist(context).log_prob(batch.mag))[selected].item())
        n=int(row.events)
        rows.append(dict(family=row.family,method=row.method,seed=seed,tied_time_nll=old,
            tied_time_nll_lower_floor=lower,tied_mark_nll=mark,primary_total=float(row.total),
            lower_floor_total=float(row.total)+(lower-old)/n,
            without_tie_total=(float(row.total)*n-old-mark)/(n-1)))
        del model;torch.cuda.empty_cache()
        print('TIED_EVENT_AUDITED',label,flush=True)
    values=pd.DataFrame(rows);values.to_csv(REPORTS/'numerical_floor_sensitivity.csv',index=False)
    grouped=values.groupby(['family','method'])[['primary_total','lower_floor_total','without_tie_total']].mean()
    etas=next(r for r in read(REPORTS/'etas/tied_event_audit.json')['rows'] if r['method']=='etas_window' and r['validation_selected'])
    comparisons=[]
    for family in study.FAMILIES:
        pairs=[('local','fine'),('etas','fine'),('etas','local')]
        if (family,'local_1600') in grouped.index and (family,'fine_1600') in grouped.index:
            pairs.append(('local_1600','fine_1600'))
        for baseline,method in pairs:
            a=grouped.loc[(family,method)]
            if baseline=='etas':before=etas['total']-a.primary_total;after=etas['without_tie_total']-a.without_tie_total
            else:
                b=grouped.loc[(family,baseline)];before=b.primary_total-a.primary_total;after=b.without_tie_total-a.without_tie_total
            comparisons.append(dict(family=family,baseline=baseline,method=method,primary_gain=float(before),without_tie_gain=float(after),sign_unchanged=bool(np.sign(before)==np.sign(after))))
    maximum=float(np.max(np.abs(values.lower_floor_total-values.primary_total)))
    c.save(REPORTS/'numerical_floor_audit.json',dict(passed=True,primary_unchanged=True,index=index,
        event_id=spec['event_id'][index],model_interval_seconds=float((spec['ns'][index]-spec['ns'][index-1])/1e9),
        primary_floor_seconds=8.64e-6,auxiliary_floor_seconds=8.64e-8,
        max_absolute_joint_nll_shift=maximum,comparisons=comparisons,rows=rows))
    report=REPORTS/'results_zh.md'
    text=report.read_text().split('\n## 同刻事件数值敏感性\n')[0]
    report.write_text(text+f'\n## 同刻事件数值敏感性\n\nSCEDC测试集有1个同刻事件展开为1微秒，小于原模型1e-10天（8.64微秒）的似然及Weibull组件内部数值下限。主表保留原设置。将测试输入和Weibull组件下限都降为1e-12天时，各模型联合NLL最大变化为{maximum:.7f} nats/事件；另对所有方法统一排除该事件进行敏感性评分。结果见numerical_floor_audit.json和numerical_floor_sensitivity.csv，不据此重选检查点。\n')
    print('NUMERICAL_FLOOR_AUDIT_COMPLETE',maximum,flush=True)


if __name__=='__main__':
    if '--etas' in sys.argv:etas_points()
    else:main()
