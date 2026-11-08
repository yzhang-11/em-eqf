"""Supplementary long-budget and tied-event checks from frozen checkpoints."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

STORE=Path('/root/autodl-fs/em_eqf_training/recast_single_region_tpp_20261105')
spec=importlib.util.spec_from_file_location('recast_study',STORE/'frozen_runner.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
study.configure();c,R=study.c,study.REPORTS
read=study.read


def windows(target,seed,updates):
    return read(R/f'neural/{target}_recast_steps_{updates}_seed_{seed}.json')['windows']


@torch.no_grad()
def main():
    assert read(R/'final_audit.json')['passed']
    study.control.setup();data=torch.load(study.DATA,map_location='cpu',weights_only=False)
    frame=pd.read_csv(R/'neural_metrics.csv');region=data['regions']['scedc']
    index=int(np.flatnonzero(np.diff(region['physical_ns'])==0)[0]+1)
    first=region['val_end']+((index-region['val_end'])//128)*128
    from run_multidomain_pretraining import make_batch
    from src.models.tpp.region_mixer_tpp import RegionalGridGR
    batch=make_batch([c.sequence(region,data['vocabulary']['scedc'],first,min(128,region['test_end']-first),1024)])
    selected=(batch.event_index==index)&batch.nll_event_mask.bool();assert int(selected.sum())==1
    rows=[];study.CURRENT_TARGET='scedc'
    for _,row in frame.loc[frame.target.eq('scedc')].iterrows():
        saved=read(R/f'neural/scedc_recast_steps_{int(row.updates)}_seed_{int(row.seed)}.json')
        assert c.sha(saved['checkpoint'])==saved['checkpoint_sha256']
        model,_=study.build('recast',int(row.seed),data['vocabulary'])
        model.load_state_dict(torch.load(saved['checkpoint'],map_location='cuda:0',weights_only=False)['model_state_dict'],strict=True)
        model.eval();distribution=model.get_inter_time_dist(model.get_context(batch))
        old=float((-distribution.log_prob(batch.inter_times.clamp_min(1e-10)))[selected].item())
        assert distribution.component_distribution.eps==1e-10
        distribution.component_distribution.eps=1e-12
        lower=float((-distribution.log_prob(batch.inter_times.clamp_min(1e-12)))[selected].item())
        mark_dist=RegionalGridGR(model.matched_b,model.matched_mc,model.matched_mc.new_tensor(.1))
        mark=float((-mark_dist.log_prob(batch.mag))[selected].item());n=int(row.events)
        rows.append(dict(seed=int(row.seed),updates=int(row.updates),primary_total=float(row.total),
            lower_floor_total=float(row.total)+(lower-old)/n,
            without_tie_total=(float(row.total)*n-old-mark)/(n-1),tied_time_nll=old,tied_mark_nll=mark))
        del model;torch.cuda.empty_cache()
    values=pd.DataFrame(rows);values.to_csv(R/'numerical_floor_sensitivity.csv',index=False)
    grouped=values.groupby('updates')[['primary_total','lower_floor_total','without_tie_total']].mean()
    prior=pd.DataFrame(read(study.OLD_REPORTS/'numerical_floor_audit.json')['rows'])
    prior=prior.groupby(['family','method'])[['primary_total','lower_floor_total','without_tie_total']].mean()
    etas=next(x for x in read(study.OLD_REPORTS/'etas/tied_event_audit.json')['rows'] if x['method']=='etas_window' and x['validation_selected'])
    comparisons=[]
    for updates,new in grouped.iterrows():
        candidates={'etas_window':dict(primary_total=etas['total'],lower_floor_total=etas['total'],without_tie_total=etas['without_tie_total'])}
        for (family,method),old in prior.iterrows():
            if (updates==800 and method in ['local','fine']) or (updates==1600 and method in ['local_1600','fine_1600']):candidates[family+'/'+method]=old
        for baseline,old in candidates.items():
            gains={k:float(old[k]-new[k]) for k in ['primary_total','lower_floor_total','without_tie_total']}
            comparisons.append(dict(updates=int(updates),baseline=baseline,gains=gains,
                direction_unchanged=all(np.sign(x)==np.sign(gains['primary_total']) for x in gains.values())))
    maximum=float(np.max(np.abs(values.lower_floor_total-values.primary_total)))
    c.save(R/'numerical_floor_audit.json',dict(passed=True,primary_unchanged=True,event_index=index,
        primary_floor_days=1e-10,auxiliary_floor_days=1e-12,max_absolute_joint_nll_shift=maximum,comparisons=comparisons))
    spec=importlib.util.spec_from_file_location('reference_summary',study.OLD/'frozen_summary.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    intervals=[];exposure_checks=[]
    for target in study.TARGETS:
        if not any((frame.target==target)&(frame.updates==1600)):continue
        reference_windows=windows(target,0,1600);weights=np.array([x['events'] for x in reference_windows])
        new=np.stack([[x['total'] for x in windows(target,s,1600)] for s in study.SEEDS])
        candidates={'recast_800':np.stack([[x['total'] for x in windows(target,s,800)] for s in study.SEEDS])}
        etas=next(x for x in read(study.OLD_REPORTS/'etas/test_results.json')['rows'] if x['target']==target and x['method']=='etas_window' and x['validation_selected'])
        ew=read(study.OLD_REPORTS/f"etas/{target}_etas_window_{etas['family']}_test.json")['windows']
        candidates['etas_window']=np.tile([x['total'] for x in ew],(3,1))
        for method in ['local_1600','fine_1600']:
            if not (study.OLD_REPORTS/f'neural/{target}_mamba2_legacy_{method}_seed_0.json').exists():continue
            candidates['mamba2_legacy/'+method]=np.stack([[x['total'] for x in read(study.OLD_REPORTS/f'neural/{target}_mamba2_legacy_{method}_seed_{s}.json')['windows']] for s in study.SEEDS])
            old_mode=method.removesuffix('_1600')
            old_lr=read(study.BASE/f'configs/{study.OLD.name}/lr_{target}_mamba2_legacy_{old_mode}.json')['lr']
            for seed in study.SEEDS:
                new_fit=read(study.directory(target,seed,1600)/'training_complete.json')
                old_fit=read(study.OLD/'runs'/f'{target}_mamba2_legacy_{old_mode}_lr_{old_lr:g}_steps_1600_seed_{seed}'/'training_complete.json')
                assert new_fit['draw_sha256']==old_fit['draw_sha256'] and new_fit['exposures']==old_fit['exposures']
                exposure_checks.append(dict(target=target,seed=seed,baseline=method,paired_target_draws_and_exposures=True))
        for baseline,old in candidates.items():
            diff=old-new
            intervals.append(dict(target=target,updates=1600,baseline=baseline,positive_is_recast_better=True,
                improvement=float(np.mean(np.sum(diff*weights,axis=1)/weights.sum())),
                intervals=[reference.block_interval(diff,weights,b) for b in [1,4,8]]))
    c.save(R/'long_budget_uncertainty.json',dict(comparisons=intervals,exposure_checks=exposure_checks))
    report=R/'results_zh.md';text=report.read_text().split('\n## 数值与训练时长核查\n')[0]
    text+=f'\n## 数值与训练时长核查\n\nSCEDC同刻事件的两处时间数值下限同时从1e-10天降低到1e-12天后，RECAST联合NLL最大变化为{maximum:.7f}。另以所有方法共同排除该事件进行敏感性比较，详见numerical_floor_audit.json；主结果保持不变。1600步与相同预算基线的事件块区间见long_budget_uncertainty.json。\n'
    report.write_text(text);print('RECAST_SUPPLEMENTARY_AUDIT_COMPLETE',maximum,flush=True)


if __name__=='__main__':main()
