"""Paired reference comparisons, causality and time-floor checks for the new default."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

STORE=Path('/root/autodl-fs/em_eqf_training/mamba3_decay_transfer_20261105')
sp=importlib.util.spec_from_file_location('default_study',STORE/'frozen_runner.py')
study=importlib.util.module_from_spec(sp);sp.loader.exec_module(study);study.configure()
c,R=study.c,study.REPORTS;read=study.read;OLD_REPORTS=study.BASE/'reports'/study.OLD.name


def new_windows(target,mode,updates):
    return [read(R/f'neural/{target}_{mode}_steps_{updates}_seed_{s}.json')['windows'] for s in study.SEEDS]


@torch.no_grad()
def main():
    assert read(R/'final_audit.json')['passed'];study.setup()
    data=torch.load(study.DATA,map_location='cpu',weights_only=False)
    from run_multidomain_pretraining import make_batch
    spec=importlib.util.spec_from_file_location('previous_summary',study.OLD/'frozen_summary.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    summary=read(R/'summary.json')['rows'];comparisons=[];comparison_rows=[];causal=[];numeric=[];refs={}
    originals=study.BASE/'reports/original_etas_tpp_20261105'
    oldsummary=read(OLD_REPORTS/'summary.json');native_rows=read(originals/'summary.json')['rows']
    for x in summary:
        comparison_rows.append(dict(target=x['target'],method='Mamba3_default/'+x['mode'],updates=x['updates'],**{k:x['metrics'][k]['mean'] for k in ['time','mag','total']}))
    for x in oldsummary['neural']:
        comparison_rows.append(dict(target=x['target'],method=x['family']+'/'+x['method'],updates=6400 if x['method']=='mixed_zero' else (1600 if x['method'].endswith('_1600') else 800),**{k:x['metrics'][k]['mean'] for k in ['time','mag','total']}))
    for x in native_rows:comparison_rows.append(dict(target=x['target'],method='ETAS_original',updates=None,**{k:x[k] for k in ['time','mag','total']}))
    for x in read(study.BASE/'reports/recast_single_region_tpp_20261105/summary.json')['recast']:
        comparison_rows.append(dict(target=x['target'],method='RECAST/local',updates=x['updates'],**{k:x['metrics'][k]['mean'] for k in ['time','mag','total']}))
    for target in study.TARGETS:
        for mode,updates in [(x['mode'],x['updates']) for x in summary if x['target']==target]:
            own=new_windows(target,mode,updates);weights=np.array([x['events'] for x in own[0]])
            new=np.array([[x['total'] for x in seed] for seed in own]);baselines={}
            ep=originals/f'{target}_test.json';refs[str(ep)]=c.sha(ep);ew=read(ep)['windows']
            baselines['ETAS_original']=np.tile([x['total'] for x in ew],(3,1))
            for family in ['mamba2_legacy','mamba3_oracle_residual']:
                oldmode=mode+('_1600' if updates==1600 else '')
                paths=[OLD_REPORTS/f'neural/{target}_{family}_{oldmode}_seed_{s}.json' for s in study.SEEDS]
                if not all(p.exists() for p in paths):continue
                for p in paths:refs[str(p)]=c.sha(p)
                baselines[family+'/'+oldmode]=np.array([[x['total'] for x in read(p)['windows']] for p in paths])
            if mode=='fine' and updates==800:
                baselines['new_default/local']=np.array([[x['total'] for x in seed] for seed in new_windows(target,'local',800)])
            if updates==1600:
                baselines['new_default/'+mode+'_800']=np.array([[x['total'] for x in seed] for seed in new_windows(target,mode,800)])
            for name,baseline in baselines.items():
                assert baseline.shape==new.shape and np.isfinite(baseline).all() and np.isfinite(new).all()
                diff=baseline-new
                comparisons.append(dict(target=target,mode=mode,updates=updates,baseline=name,positive_is_new_default_better=True,
                    improvement=float(np.mean(np.sum(diff*weights,axis=1)/weights.sum())),intervals=[ref.block_interval(diff,weights,b) for b in [1,4,8]]))
        # A trained model must not use future magnitude/gap values for its prefix.
        saved=read(R/f'neural/{target}_fine_steps_800_seed_0.json');model,_=study.build(study.FAMILY,0,data['vocabulary'])
        model.load_state_dict(torch.load(saved['checkpoint'],map_location='cuda:0',weights_only=False)['model_state_dict'],strict=True);model.eval()
        d=data['regions'][target];batch=make_batch([c.sequence(d,data['vocabulary'][target],256,128,128)])
        changed=copy.deepcopy(batch);start=int(torch.nonzero(batch.nll_event_mask.bool())[0,1])
        changed.mag[:,start:]+=.5;changed.inter_times[:,start:]*=1.5
        a=model.get_context(batch);b=model.get_context(changed);error=float((a[:,:start+1]-b[:,:start+1]).abs().max());assert error<1e-6
        causal.append(dict(target=target,future_magnitude_and_gap_prefix_error=error))
        del model;torch.cuda.empty_cache()
    # Preserve the primary scores; quantify the one tied SCEDC event separately.
    d=data['regions']['scedc'];index=int(np.flatnonzero(np.diff(d['physical_ns'])==0)[0]+1)
    first=d['val_end']+((index-d['val_end'])//128)*128
    batch=make_batch([c.sequence(d,data['vocabulary']['scedc'],first,min(128,d['test_end']-first),1024)])
    selected=(batch.event_index==index)&batch.nll_event_mask.bool();assert int(selected.sum())==1
    frame=pd.read_csv(R/'neural_metrics.csv')
    for _,row in frame.loc[frame.target.eq('scedc')].iterrows():
        saved=read(R/f'neural/scedc_{row["mode"]}_steps_{int(row.updates)}_seed_{int(row.seed)}.json')
        model,_=study.build(study.FAMILY,int(row.seed),data['vocabulary']);model.load_state_dict(torch.load(saved['checkpoint'],map_location='cuda:0',weights_only=False)['model_state_dict'],strict=True);model.eval()
        context=model.get_context(batch);dist=model.get_inter_time_dist(context)
        old=float((-dist.log_prob(batch.inter_times.clamp_min(1e-10)))[selected].item());dist.component_distribution.eps=1e-12
        lower=float((-dist.log_prob(batch.inter_times.clamp_min(1e-12)))[selected].item())
        mark=float((-model.get_magnitude_dist(context).log_prob(batch.mag))[selected].item());n=int(row.events)
        numeric.append(dict(mode=row['mode'],updates=int(row.updates),seed=int(row.seed),primary_total=float(row.total),lower_floor_total=float(row.total)+(lower-old)/n,without_tie_total=(float(row.total)*n-old-mark)/(n-1)))
        del model;torch.cuda.empty_cache()
    pd.DataFrame(comparison_rows).to_csv(R/'comparison.csv',index=False);pd.DataFrame(numeric).to_csv(R/'numerical_floor_sensitivity.csv',index=False)
    c.save(R/'paired_uncertainty.json',dict(comparisons=comparisons,reference_file_sha256=refs))
    maximum=max(abs(x['primary_total']-x['lower_floor_total']) for x in numeric)
    c.save(R/'supplementary_audit.json',dict(passed=True,causality=causal,time_floor_max_joint_shift=maximum,primary_scores_unchanged=True))
    text=(R/'results_zh.md').read_text().split('\n## 同窗参考与补充检查\n')[0]
    text+='\n## 同窗参考与补充检查\n\n|地区|方法|预算|时间NLL|震级NLL|联合NLL|\n|---|---|---|---|---|---|\n'
    for row in comparison_rows:text+=f"|{row['target']}|{row['method']}|{row['updates']}|{row['time']:.5f}|{row['mag']:.5f}|{row['total']:.5f}|\n"
    text+=f'\n训练后未来震级/间隔扰动的历史前缀误差均低于1e-6；SCEDC同刻事件的似然下限敏感性最大为{maximum:.7f} nats/事件。主评分保持不变。原实现ETAS作为主对照，参数与预算见独立original_etas_tpp报告。\n新旧Mamba3同时改变MIMO/状态尺寸、时间机制、输入、输出参数化和残差，本表比较完整版本，不能单独归因于衰减调制。新默认去掉残差后需以本轮新训练结果为准，不沿用历史残差开启的完整序列结果。\n'
    (R/'results_zh.md').write_text(text);print('DEFAULT_MAMBA3_SUPPLEMENTARY_COMPLETE',flush=True)


if __name__=='__main__':main()
