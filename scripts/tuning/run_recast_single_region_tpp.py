"""Original RECAST on the frozen ChuanDian/SCEDC TPP windows."""
import argparse
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
TAG='recast_single_region_tpp_20261105'
BASE=ROOT/'experiments/revision/multiregion_pretraining'
STORE=Path('/root/autodl-fs/em_eqf_training')/TAG
OLD=STORE.parent/'chuandian_scedc_tpp_20261105'
OLD_REPORTS=BASE/'shared/reports'/OLD.name
REPORTS,CONFIGS,RUNS=[BASE/k/TAG for k in ['shared/reports','configs','runs']]
DATA=OLD/'targets.pt'
spec=importlib.util.spec_from_file_location('matched_tpp_control',OLD/'frozen_runner.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
c=control.c
TARGETS=['chuandian','scedc'];SEEDS=[0,1,2];LRS=[1e-4,3e-4,1e-3]
CURRENT_TARGET=None


def read(path):return json.loads(Path(path).read_text())


def mle_b(magnitude,mc):
    mean=float(np.rint((magnitude-mc)/.1).mean());bins=round((10-mc)/.1)
    low,high=.2,2.
    for _ in range(80):
        b=(low+high)/2;rate=b*np.log(10)*.1
        expected=1/np.expm1(rate)-bins/np.expm1(bins*rate)
        if expected>mean:low=b
        else:high=b
    return (low+high)/2


def build(family,seed,vocabulary):
    assert family=='recast' and CURRENT_TARGET in TARGETS
    from src.models.tpp.recurrent.model_v1 import RecurrentTPP
    from src.models.tpp.region_mixer_tpp import RegionalGridGR

    class MatchedRecast(RecurrentTPP):
        def __init__(self,args):
            super().__init__(args,device=torch.device('cuda:0'))
            self.register_buffer('matched_b',torch.tensor(args.richter_b_mle,dtype=torch.float64,device=self.device))
            self.register_buffer('matched_mc',torch.tensor(args.mag_completeness,dtype=torch.float64,device=self.device))

        def _run_rnn(self,features,hidden_state=None):
            # Cast only encoder features; the event clock and scoring mask stay
            # float64, just as in the Mamba control pipeline.
            return super()._run_rnn(features.to(self.rnn.weight_ih_l0),hidden_state)

        def nll_loss(self,batch,*,reduction='per_event'):
            assert reduction=='per_event'
            time_loss=super().nll_loss(batch,reduction=reduction,return_dict=True)['time']
            mark=RegionalGridGR(self.matched_b,self.matched_mc,self.matched_mc.new_tensor(.1))
            mask=batch.nll_event_mask.bool();n=mask.sum(-1)
            mark_loss=-mark.log_prob(batch.mag,mask=mask).sum(-1)/n
            return dict(time=time_loss,mag=mark_loss,total=time_loss+mark_loss)

    cfg=copy.deepcopy(read(CONFIGS/'study_plan.json')['model_configs'][CURRENT_TARGET])
    torch.manual_seed(seed)
    model=MatchedRecast(OmegaConf.create(cfg))
    torch.manual_seed(seed+1000)
    return model,cfg


def configure():
    # Reuse the exact immutable optimizer, batches, sampling and checkpoint
    # selection code of the previous study; no shared source is edited.
    control.DATA=DATA;control.RUNS=RUNS;control.REPORTS=REPORTS
    control.CONFIGS=CONFIGS;control.STORE=STORE;control.build=build


def initialize():
    for p in [STORE,REPORTS,CONFIGS,STORE/'runs']:p.mkdir(parents=True,exist_ok=True)
    if not RUNS.exists():RUNS.symlink_to(STORE/'runs',target_is_directory=True)
    path=CONFIGS/'study_plan.json'
    if path.exists():
        plan=read(path);assert c.sha(STORE/'frozen_runner.py')==plan['runner_sha256']
        return
    assert read(OLD_REPORTS/'final_audit.json')['passed']
    template=ROOT/'checkpoints/rtpp_20260506-173356/config.yaml'
    shutil.copy2(template,CONFIGS/'original_recast_template.yaml')
    shutil.copy2(Path(__file__),STORE/'frozen_runner.py')
    data=torch.load(DATA,map_location='cpu',weights_only=False)
    configs={}
    for target,d in data['regions'].items():
        cfg=OmegaConf.to_container(OmegaConf.load(template),resolve=True)
        b=mle_b(d['magnitude'][256:d['train_end']],d['metadata'][0])
        assert abs(b-read(OLD_REPORTS/f'etas/{target}_window_classic.json')['parameters']['b'])<1e-12
        cfg.update(d_model=32,num_rnn_layers=1,rnn_type='GRU',rnn_dropout=.3,num_components=32,
            dataset='ChuanDian' if target=='chuandian' else 'SCEDC',Mc=d['metadata'][0],
            rnn_chunk_len=4096,bg_model=None,scale_range='positive',loss_reduction='per_event',
            tau_mean=d['metadata'][2],mag_mean=float(d['magnitude'][:256].mean()),
            time_max=float(np.diff(d['ns'][:d['train_end']]).max()/86400e9),
            richter_b_mle=b,mag_completeness=d['metadata'][0],use_double_precision=False)
        configs[target]=cfg
    references={str(OLD/'frozen_runner.py'):c.sha(OLD/'frozen_runner.py'),
                str(OLD/'frozen_summary.py'):c.sha(OLD/'frozen_summary.py')}
    references.update({str(OLD_REPORTS/name):c.sha(OLD_REPORTS/name) for name in ['summary.json','neural_metrics.csv','etas/test_results.json','data.json']})
    plan=dict(protocol=TAG,model='Original RECAST rtpp;1-layer GRU32;32-component legacy Weibull;no background model',
        model_configs=configs,targets=TARGETS,seeds=SEEDS,learning_rates=LRS,
        runner_sha256=c.sha(STORE/'frozen_runner.py'),data_path=str(DATA),data_sha256=c.sha(DATA),
        original_template=str(template),original_template_sha256=c.sha(template),references=references,
        updates=800,optimizer='Same frozen AdamW(weight_decay=.01),clip1,warmup10%+cosine5% floor as Mamba controls.',
        learning_rate_selection='Three-rate seed0 validation search, including the original1e-3; fixed selected rate for seeds1/2. Same search count as Mamba, different range.',
        history=1024,training_targets=2048,evaluation_targets=128,calibration_events=256,
        normalization='Same early256 training events: mean inter-event time and mean magnitude; native RECAST centering retained.',
        marks='Native RECAST trains time only. Add fixed train-MLE GR0.1-bin mark likelihood for a comparable joint NLL;constant term has no effect on gradients.',
        precision='FP32 parameters/features;float64 untouched event clock;encoder-only feature cast;compared with nativeFP64 in preflight.',
        long_guard='If selected seed0 best_step=800 and validation600-validation800>0.002, separately train1600 steps for all3seeds before any new test scoring.',
        limitations=['Previously studied test years;exploratory follow-up.','Original RECAST is GRU1x32, not a parameter-matched Mamba2x64 ablation.','Normalization and windows match this benchmark rather than old full-history training.'])
    c.save(path,plan);c.save(REPORTS/'plan.json',plan)


def preflight():
    global CURRENT_TARGET
    control.setup();data=torch.load(DATA,map_location='cpu',weights_only=False);rows=[]
    from src.models.tpp.recurrent.model_v1 import RecurrentTPP
    from run_multidomain_pretraining import make_batch
    for target in TARGETS:
        CURRENT_TARGET=target;c._TRAIN_CACHE.clear()
        model,cfg=build('recast',0,data['vocabulary']);batch=c.training_batch(data,target,0,'long')
        before=batch.inter_times.clone();started=time.time()
        losses=model.nll_loss(batch);loss=losses['total'].mean();loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert torch.equal(before,batch.inter_times) and batch.inter_times.dtype==torch.float64
        assert int(batch.nll_event_mask.sum())==data['regions'][target]['block_targets'][0]
        model.eval()
        probe=make_batch([c.sequence(data['regions'][target],data['vocabulary'][target],256,128,128)])
        native=RecurrentTPP(OmegaConf.create(cfg),device=torch.device('cuda:0')).double().eval()
        state={k:v for k,v in model.state_dict().items() if k not in ['matched_b','matched_mc']}
        native.load_state_dict(state,strict=True)
        with torch.no_grad():
            error=float((native.nll_loss(probe,reduction='per_event',return_dict=True)['time']-model.nll_loss(probe)['time']).abs().max())
            assert error<5e-4,(target,error)
            changed=copy.deepcopy(probe);first=int(torch.nonzero(probe.nll_event_mask.bool())[0,1])
            changed.mag[:,first:]+=.7;changed.inter_times[:,first:]*=1.5
            original=model.get_context(probe);perturbed=model.get_context(changed)
            causal_error=float((original[:,:first+1]-perturbed[:,:first+1]).abs().max())
            assert causal_error<1e-6,(target,causal_error)
        rows.append(dict(target=target,finite_loss=float(loss.detach()),finite_gradient=True,
            raw_clock_unchanged=True,scored_events=int(batch.nll_event_mask.sum()),
            parameters=sum(p.numel() for p in model.parameters()),native_fp64_nll_error=error,
            causal_prefix_error=causal_error,seconds=time.time()-started))
        del native,model,batch;torch.cuda.empty_cache()
        print('RECAST_PREFLIGHT',rows[-1],flush=True)
    c._TRAIN_CACHE.clear();c.save(REPORTS/'preflight.json',dict(passed=True,checks=rows))


def fit(target,seed,updates=800,grid=False):
    global CURRENT_TARGET
    CURRENT_TARGET=target
    selection=CONFIGS/f'lr_{target}_recast_local.json'
    if grid:
        trials=[control.fit(target,'recast',seed,'local',lr,updates) for lr in LRS]
        best=min(trials,key=lambda x:x['validation']['macro']['total'])
        c.save(selection,dict(lr=best['lr'],validation={str(x['lr']):x['validation']['macro'] for x in trials},test_used=False))
    else:control.fit(target,'recast',seed,'local',read(selection)['lr'],updates)


def directory(target,seed,updates=800):
    lr=read(CONFIGS/f'lr_{target}_recast_local.json')['lr']
    return RUNS/f'{target}_recast_local_lr_{lr:g}_steps_{updates}_seed_{seed}'


def jobs_run(jobs,workers):
    jobs=list(jobs);active=[]
    while jobs or active:
        while jobs and len(active)<workers:
            target,seed,updates,grid=jobs.pop(0);label=f'{target}_seed_{seed}_steps_{updates}'
            log=(REPORTS/f'{label}.log').open('a')
            command=[sys.executable,'-u',str(STORE/'frozen_runner.py'),'--worker','--target',target,'--seed',str(seed),'--updates',str(updates)]
            if grid:command.append('--grid')
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);active.append((p,log,label));print('START',label,p.pid,flush=True)
        for item in active[:]:
            p,log,label=item;status=p.poll()
            if status is not None:
                active.remove(item);log.close()
                if status:
                    for other,_,_ in active:other.terminate()
                    raise RuntimeError(f'{label} failed: {status}')
        time.sleep(2)


def master(workers):
    initialize()
    if not (REPORTS/'preflight.json').exists():preflight()
    jobs_run([(t,0,800,True) for t in TARGETS],workers)
    jobs_run([(t,s,800,False) for t,s in itertools.product(TARGETS,[1,2])],workers)
    guarded=[]
    for target in TARGETS:
        result=read(directory(target,0)/'training_complete.json')
        curve={x['step']:x['validation']['macro']['total'] for x in result['history']}
        if result['best_step']==800 and curve[600]-curve[800]>.002:guarded.append(target)
    c.save(CONFIGS/'budget_guard.json',dict(triggered=guarded,test_used=False))
    jobs_run([(t,s,1600,False) for t,s in itertools.product(guarded,SEEDS)],workers)
    c.save(REPORTS/'training_complete.json',dict(complete=True,test_scored=False))
    print('RECAST_TRAINING_COMPLETE',flush=True)
    finalize()


def finalize():
    global CURRENT_TARGET
    assert read(REPORTS/'training_complete.json')['complete']
    plan=read(CONFIGS/'study_plan.json');assert c.sha(DATA)==plan['data_sha256']
    assert c.sha(STORE/'frozen_runner.py')==plan['runner_sha256']
    assert all(c.sha(path)==digest for path,digest in plan['references'].items())
    control.setup();data=torch.load(DATA,map_location='cpu',weights_only=False)
    guarded=read(CONFIGS/'budget_guard.json')['triggered'];rows=[];audits=[];windows={}
    for target,seed in itertools.product(TARGETS,SEEDS):
        CURRENT_TARGET=target
        for updates in [800]+([1600] if target in guarded else []):
            folder=directory(target,seed,updates);saved=read(folder/'training_complete.json')
            assert saved['dataset_sha256']==plan['data_sha256']
            assert c.sha(folder/'best_model.pth')==saved['checkpoint_sha256']
            model,cfg=build('recast',seed,data['vocabulary']);state=torch.load(folder/'best_model.pth',map_location='cuda:0',weights_only=False)
            assert cfg==saved['backbone']==state['backbone']
            model.load_state_dict(state['model_state_dict'],strict=True)
            val,_=c.evaluate(model,{target:c.evaluation_pools(data,'val',1024)[target]})
            error=max(abs(val['regions'][target][k]-saved['validation']['regions'][target][k]) for k in ['time','mag','total'])
            assert error<5e-5,(target,seed,updates,error)
            metrics,window=c.evaluate(model,{target:c.evaluation_pools(data,'test',1024)[target]})
            value=metrics['regions'][target];count=sum(x['events'] for x in window[target])
            assert count==data['regions'][target]['test_end']-data['regions'][target]['val_end']
            assert abs(value['time']+value['mag']-value['total'])<2e-5
            if updates==800:
                prior_lr=read(BASE/f'configs/{OLD.name}/lr_{target}_mamba2_legacy_local.json')['lr']
                prior=read(OLD/'runs'/f'{target}_mamba2_legacy_local_lr_{prior_lr:g}_steps_800_seed_{seed}'/'training_complete.json')
                assert prior['draw_sha256']==saved['draw_sha256'] and prior['exposures']==saved['exposures']
            label=f'{target}_recast_steps_{updates}_seed_{seed}'
            c.save(REPORTS/f'neural/{label}.json',dict(validation=val['regions'][target],test=value,windows=window[target],
                checkpoint=str(folder/'best_model.pth'),checkpoint_sha256=saved['checkpoint_sha256']))
            rows.append(dict(target=target,seed=seed,updates=updates,lr=saved['lr'],best_step=state['step'],events=count,**value))
            windows[(target,seed,updates)]=window[target]
            audits.append(dict(target=target,seed=seed,updates=updates,validation_replay_error=error,strict_loading=True,paired_target_draws_verified=updates==800))
            del model;torch.cuda.empty_cache();print('RECAST_SCORED',label,value,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(REPORTS/'neural_metrics.csv',index=False)
    groups=[]
    for (target,updates),group in frame.groupby(['target','updates']):
        groups.append(dict(target=target,updates=int(updates),metrics={k:dict(mean=float(group[k].mean()),std=float(group[k].std(ddof=1))) for k in ['time','mag','total']}))
    c.save(REPORTS/'summary.json',dict(recast=groups,reference=read(OLD_REPORTS/'summary.json')))
    spec=importlib.util.spec_from_file_location('reference_scoring',OLD/'frozen_summary.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    intervals=[]
    for target in TARGETS:
        weights=np.array([x['events'] for x in windows[(target,0,800)]])
        recast=np.stack([[x['total'] for x in windows[(target,s,800)]] for s in SEEDS])
        etas=next(x for x in read(OLD_REPORTS/'etas/test_results.json')['rows'] if x['target']==target and x['method']=='etas_window' and x['validation_selected'])
        ew=read(OLD_REPORTS/f"etas/{target}_etas_window_{etas['family']}_test.json")['windows']
        candidates={'etas_window':np.tile([x['total'] for x in ew],(3,1))}
        for family,method in itertools.product(['mamba2_legacy','mamba3_oracle_residual'],['local','fine']):
            candidates[family+'/'+method]=np.stack([[x['total'] for x in read(OLD_REPORTS/f'neural/{target}_{family}_{method}_seed_{s}.json')['windows']] for s in SEEDS])
        for baseline,values in candidates.items():
            diff=values-recast
            intervals.append(dict(target=target,updates=800,baseline=baseline,positive_is_recast_better=True,
                improvement=float(np.mean(np.sum(diff*weights,axis=1)/weights.sum())),intervals=[reference.block_interval(diff,weights,b) for b in [1,4,8]]))
    c.save(REPORTS/'paired_uncertainty.json',dict(comparisons=intervals))
    c.save(REPORTS/'final_audit.json',dict(passed=True,primary_runs=6,scored_runs=len(rows),
        training_fits=len(list(RUNS.glob('*/training_complete.json'))),checks=audits,
        dataset_sha256=plan['data_sha256'],reference_hashes_unchanged=True,original_recast_core_unchanged=True))
    lines=['# 同一窗口的RECAST单区域TPP对照','','两地区均沿用上一轮冻结目录、Mc、日期划分、1024历史及128目标评分窗口；神经模型3种子均值，单位nats/事件，越低越好。',
        'RECAST采用返修基线原版rtpp：单层GRU32、32分量Weibull、dropout0.3、固定训练MLE的GR震级分布。时间项直接调用原实现；震级项按相同0.1分箱独立补算。原代码total只包含时间项，因此没有直接把它当联合NLL。',
        '编码器特征转FP32，原始事件时钟保留FP64；已与原生FP64前向和未来扰动因果探针核对。没有改动共享模型源码。',
        '800步主预算与Mamba对照相同，目标抽样和曝光逐种子核验。AdamW、权重衰减、学习率调度和梯度裁剪沿用上一轮；3学习率搜索为1e-4/3e-4/1e-3（包含RECAST原默认），搜索次数与Mamba相同、范围不同。1600步仅为验证触发的单列敏感性。',
        'RECAST原架构为1层32维，Mamba为2层64维；此处匹配数据和窗口，并非只替换编码器的等参数消融。测试年份此前已有研究使用，本轮为探索性补充。','','|地区|方法|时间NLL|震级NLL|联合NLL|','|---|---|---|---|---|']
    for item in groups:
        m=item['metrics'];fmt=lambda k:f"{m[k]['mean']:.5f} ± {m[k]['std']:.5f}"
        lines.append(f"|{item['target']}|RECAST/{item['updates']}步|{fmt('time')}|{fmt('mag')}|{fmt('total')}|")
    prior=read(OLD_REPORTS/'summary.json')
    for item in prior['neural']:
        if item['method']=='mixed_zero':continue
        m=item['metrics'];lines.append(f"|{item['target']}|{item['family']}/{item['method']}|{m['time']['mean']:.5f}|{m['mag']['mean']:.5f}|{m['total']['mean']:.5f}|")
    for item in prior['etas']:
        if item['method']=='etas_window' and item['validation_selected']:
            lines.append(f"|{item['target']}|ETAS同窗/{item['family']}|{item['time']:.5f}|{item['mag']:.5f}|{item['total']:.5f}|")
    lines+=['','配置及归一化统计见plan.json；各次拟合见training_manifest.csv；逐种子与最佳步数见neural_metrics.csv；成对事件块区间见paired_uncertainty.json。','原基线报告和全部原数据保持原状；参考指标来源及SHA-256见plan.json的references。']
    (REPORTS/'results_zh.md').write_text('\n'.join(lines)+'\n');print('RECAST_STUDY_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--run',action='store_true');parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--grid',action='store_true')
    parser.add_argument('--target',choices=TARGETS);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--updates',type=int,default=800);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();configure()
    if args.initialize:initialize()
    elif args.preflight:preflight()
    elif args.run:master(args.workers)
    elif args.finalize:finalize()
    elif args.worker:fit(args.target,args.seed,args.updates,args.grid)
    else:parser.error('Choose an action')
