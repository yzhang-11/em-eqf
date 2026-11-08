"""Mamba3 default: decay-only, native phase, masked direct dt, legacy head, no residual."""
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
import torch

ROOT=Path('/root/autodl-tmp/em_eqf');TAG='mamba3_decay_transfer_20261105'
BASE=ROOT/'experiments/revision/multiregion_pretraining';STORE=Path('/root/autodl-fs/em_eqf_training')/TAG
FROZEN=STORE/'frozen_project';OLD=STORE.parent/'chuandian_scedc_tpp_20261105'
REPORTS,CONFIGS,RUNS=[BASE/x/TAG for x in ['reports','configs','runs']]
DATA=OLD/'targets.pt';FAMILY='mamba3_decay_native';TARGETS=['chuandian','scedc'];SEEDS=[0,1,2];LRS=[3e-5,1e-4,3e-4]
spec=importlib.util.spec_from_file_location('previous_control',OLD/'frozen_runner.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);c=control.c


def read(p):return json.loads(Path(p).read_text())


def configure():
    c.FROZEN=FROZEN;c.REPORTS=REPORTS
    control.STORE=STORE;control.REPORTS=REPORTS;control.CONFIGS=CONFIGS;control.RUNS=RUNS;control.DATA=DATA
    control.setup=setup;control.build=build
    control.source_data=lambda target:OLD/f'sources_before_{target}_training_end.pt'


def setup():
    c.bootstrap();torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True


def build(family,seed,vocabulary):
    assert family==FAMILY
    from run_multidomain_pretraining import build_model
    cfg=copy.deepcopy(read(CONFIGS/'study_plan.json')['backbone'])
    model=build_model('stats_film',seed,vocabulary,cfg)
    assert model.region_embedding is None and model.inter_time_decoder.parametrization=='legacy'
    blocks=[m for m in model.modules() if hasattr(m,'use_residual')];assert blocks and all(not m.use_residual for m in blocks)
    mixers=[m for m in model.modules() if hasattr(m,'time_update_mode')]
    assert len(mixers)==2 and all(m.time_update_mode=='decay_only' and not m.use_event_time_phase and m.is_mimo and m.mimo_rank==2 for m in mixers)
    adapter=model.base_model.input_adapter
    assert adapter.provide_physical_dt and adapter.mask_input_features==('log_inter_times',)
    return model,cfg


def initialize():
    for p in [STORE,REPORTS,CONFIGS,STORE/'runs']:p.mkdir(parents=True,exist_ok=True)
    if not RUNS.exists():RUNS.symlink_to(STORE/'runs',target_is_directory=True)
    if (CONFIGS/'study_plan.json').exists():
        assert c.sha(STORE/'frozen_runner.py')==read(CONFIGS/'study_plan.json')['runner_sha256'];return
    files=list((ROOT/'src').rglob('*.py'))+[ROOT/x for x in [
        'scripts/run/run_multidomain_pretraining.py','scripts/run/run_multiregion_pretraining.py',
        'experiments/revision/main/configs/pretrain_azdx_main_seed_0.yaml',
        'config/mixer_tpp_mamba3_event_time.yaml','src/models/mamba/README_mamba3_event_time.md','src/models/mamba/README_mixer_backbones.md']]
    manifest={}
    for source in files:
        relative=source.relative_to(ROOT);dest=FROZEN/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);manifest[str(relative)]=c.sha(dest)
    c.save(REPORTS/'frozen_source.json',manifest);shutil.copy2(Path(__file__),STORE/'frozen_runner.py')
    previous=read(BASE/f'configs/{OLD.name}/study_plan.json')
    cfg=copy.deepcopy(previous['backbones']['mamba2_legacy'])
    profile=OmegaConf.to_container(OmegaConf.load(ROOT/'config/mixer_tpp_mamba3_event_time.yaml'),resolve=True)
    cfg['mixer_model_config']['ssm_cfg']=profile['mixer_model_config']['ssm_cfg']
    cfg['mixer_model_config']['use_residual']=False
    cfg.update(mixer_time_parametrization='legacy',provide_physical_dt=True,mask_input_features=['log_inter_times'])
    plan=dict(protocol=TAG,family=FAMILY,backbone=cfg,seeds=SEEDS,targets=TARGETS,
        data_sha256=c.sha(DATA),source_data=previous['source_data'],runner_sha256=c.sha(STORE/'frozen_runner.py'),
        control_runner_sha256=c.sha(OLD/'frozen_runner.py'),default_profile_sha256=c.sha(ROOT/'config/mixer_tpp_mamba3_event_time.yaml'),
        source_updates=6400,target_updates=800,learning_rates=LRS,source_learning_rate=3e-4,
        method='stats_film; no regionID embedding; whole TPP transfer',
        architecture='Mamba3EventTime MIMO rank2,state64,head32;2layers64;decay_only;native phase;no directlogdt;legacy head;NO residual',
        sampling_optimizer='Exact prior frozen fit() implementation: same data,draws,AdamW.01,clip1,warmup10%,cosine5%floor.',
        source_selection='Separate as-of8-region source cohort per target;source validation macro joint NLL;target regions and overlapping NCEDC excluded.',
        target_selection='3rates on seed0,local/fine separately;fixed selected rate on seeds1/2;target validation best checkpoint.',
        long_guard='If seed0 selected best step800 and val600-val800>.002,run separate1600-step sensitivity all3seeds before test.',
        score='time and0.1-bin magnitude joint NLL/event;1024history+128target evaluation;first256target training events calibration only.',
        limitations=['Historical test periods;exploratory follow-up.','MIMO/state/head/update/input/residual settings differ from earlier SISO+Oracle;not an isolated architecture ablation.','Pretraining adds compute beyond target updates.'])
    c.save(CONFIGS/'study_plan.json',plan);c.save(REPORTS/'plan.json',plan)


def preflight():
    setup();data=torch.load(DATA,map_location='cpu',weights_only=False)
    from run_multidomain_pretraining import make_batch
    from src.models.tpp import region_mixer_tpp as regional
    # Check the previous regional adapter with both new controls disabled.
    namespace=dict(vars(regional));old_text=(STORE/'before/src/models/tpp/region_mixer_tpp.py').read_text()
    exec(compile(old_text,'regional_adapter_before.py','exec'),namespace)
    old_adapter=namespace['RegionalBatchAdapter']
    checks=[]
    for target in TARGETS:
        c._TRAIN_CACHE.clear();model,cfg=build(FAMILY,0,data['vocabulary']);batch=c.training_batch(data,target,0,'long')
        adapter=model.base_model.input_adapter;result=adapter(batch);names=adapter.features_input_keys;idx=names.index('log_inter_times')
        assert torch.count_nonzero(result['features'][...,idx])==0
        expected=torch.where(batch.input_mask.bool(),batch.inter_times/batch.region_params[:,0,2:3],torch.ones_like(batch.inter_times))
        torch.testing.assert_close(result['physical_dt'],expected,rtol=0,atol=0)
        neutral=OmegaConf.create(cfg);neutral.mask_input_features=[];neutral.provide_physical_dt=False
        before=old_adapter(neutral)(batch);after=regional.RegionalBatchAdapter(neutral)(batch)
        assert before.keys()==after.keys() and all(torch.equal(before[k],after[k]) for k in before)
        clock=batch.inter_times.clone();model.train();loss=model.nll_loss(batch,reduction='per_event')['total'].mean();loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        projection=next(m for m in model.base_model.encoder.input_proj.modules() if isinstance(m,torch.nn.Linear) and m.in_features==2)
        assert torch.count_nonzero(projection.weight.grad[:,idx])==0
        assert torch.equal(clock,batch.inter_times)
        checks.append(dict(target=target,parameters=sum(p.numel() for p in model.parameters()),finite_loss=float(loss.detach()),
            direct_dt_zero=True,dt_projection_gradient_zero=True,physical_dt_exact=True,legacy_adapter_unchanged=True,
            raw_clock_unchanged=True,scored_events=int(batch.nll_event_mask.sum()),oracle=False,residual=False))
        del model;torch.cuda.empty_cache()
    # Mixed regional rows must use their own tau, not the global placeholder.
    items=[c.sequence(d,data['vocabulary'][target],256,32,128) for target,d in data['regions'].items()]
    model,_=build(FAMILY,0,data['vocabulary']);batch=make_batch(items);value=model.base_model.input_adapter(batch)
    expected=torch.where(batch.input_mask.bool(),batch.inter_times/batch.region_params[:,0,2:3],torch.ones_like(batch.inter_times))
    torch.testing.assert_close(value['physical_dt'],expected,rtol=0,atol=0)
    c._TRAIN_CACHE.clear();c.save(REPORTS/'preflight.json',dict(passed=True,checks=checks,mixed_region_tau_verified=True))
    print('DECAY_DEFAULT_PREFLIGHT_PASSED',checks,flush=True)


def selected_dir(target,mode,seed,updates=800):
    lr=read(CONFIGS/f'lr_{target}_{FAMILY}_{mode}.json')['lr']
    return RUNS/f'{target}_{FAMILY}_{mode}_lr_{lr:g}_steps_{updates}_seed_{seed}'


def worker(phase,target,seed,updates):
    if phase=='source':control.fit(target,FAMILY,seed,'source',3e-4,6400);return
    for mode in ['local','fine']:
        path=CONFIGS/f'lr_{target}_{FAMILY}_{mode}.json'
        if phase=='guard':continue
        if seed==0:
            trials=[control.fit(target,FAMILY,seed,mode,lr) for lr in LRS];best=min(trials,key=lambda x:x['validation']['macro']['total'])
            c.save(path,dict(lr=best['lr'],test_used=False,validation={str(x['lr']):x['validation']['macro'] for x in trials}))
        else:control.fit(target,FAMILY,seed,mode,read(path)['lr'])


def jobs_run(jobs,workers):
    jobs=list(jobs);active=[]
    while jobs or active:
        while jobs and len(active)<workers:
            phase,target,seed,mode=jobs.pop(0);label=f'{phase}_{target}_seed_{seed}'+('_'+mode if mode else '')
            log=(REPORTS/f'{label}.log').open('a')
            command=[sys.executable,'-u',str(STORE/'frozen_runner.py'),'--worker',phase,'--target',target,'--seed',str(seed)]
            if mode:command+=['--mode',mode]
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);active.append((p,log,label));print('START',label,p.pid,flush=True)
        for item in active[:]:
            p,log,label=item;status=p.poll()
            if status is not None:
                active.remove(item);log.close()
                if status:
                    for other,_,_ in active:other.terminate()
                    raise RuntimeError(f'{label} failed:{status}')
        time.sleep(2)


def master(workers):
    initialize()
    if not (REPORTS/'preflight.json').exists():preflight()
    jobs_run([('source',t,s,None) for t,s in itertools.product(TARGETS,SEEDS)],workers)
    jobs_run([('target',t,0,None) for t in TARGETS],workers)
    jobs_run([('target',t,s,None) for t,s in itertools.product(TARGETS,[1,2])],workers)
    guarded=[]
    for target,mode in itertools.product(TARGETS,['local','fine']):
        d=read(selected_dir(target,mode,0)/'training_complete.json');curve={x['step']:x['validation']['macro']['total'] for x in d['history']}
        if d['best_step']==800 and curve[600]-curve[800]>.002:guarded.append(dict(target=target,mode=mode))
    c.save(CONFIGS/'budget_guard.json',dict(triggered=guarded,test_used=False))
    jobs_run([('guard',x['target'],seed,x['mode']) for x,seed in itertools.product(guarded,SEEDS)],workers)
    c.save(REPORTS/'training_complete.json',dict(complete=True,test_scored=False));print('DECAY_DEFAULT_TRAINING_COMPLETE',flush=True)
    finalize()


def finalize():
    assert read(REPORTS/'training_complete.json')['complete'];setup();plan=read(CONFIGS/'study_plan.json')
    assert c.sha(DATA)==plan['data_sha256'] and c.sha(STORE/'frozen_runner.py')==plan['runner_sha256']
    assert c.sha(OLD/'frozen_runner.py')==plan['control_runner_sha256']
    data=torch.load(DATA,map_location='cpu',weights_only=False);guarded=read(CONFIGS/'budget_guard.json')['triggered'];rows=[];audits=[]
    for target,seed in itertools.product(TARGETS,SEEDS):
        source=control.source_run(target,FAMILY,seed);src=read(source/'training_complete.json')
        assert src['dataset_sha256']==plan['source_data'][target]['sha256']
        assert c.sha(control.source_data(target))==src['dataset_sha256']
        model,_=build(FAMILY,seed,data['vocabulary']);model.load_state_dict(torch.load(source/'best_model.pth',map_location='cuda:0',weights_only=False)['model_state_dict'],strict=True)
        source_state=c.state_hash(model);sd=torch.load(control.source_data(target),map_location='cpu',weights_only=False)
        replay,_=c.evaluate(model,c.evaluation_pools(sd,'val',1024));error=max(abs(replay['regions'][r][k]-src['validation']['regions'][r][k]) for r in replay['regions'] for k in ['time','mag','total']);assert error<5e-5
        del model;torch.cuda.empty_cache()
        modes=[('local',800),('mixed_zero',6400),('fine',800)]+[(x['mode'],1600) for x in guarded if x['target']==target]
        for mode,updates in modes:
            folder=source if mode=='mixed_zero' else selected_dir(target,mode,seed,updates)
            saved=read(folder/'training_complete.json');state=torch.load(folder/'best_model.pth',map_location='cuda:0',weights_only=False)
            assert c.sha(folder/'best_model.pth')==saved['checkpoint_sha256']
            model,cfg=build(FAMILY,seed,data['vocabulary']);assert cfg==state['backbone']==saved['backbone'];model.load_state_dict(state['model_state_dict'],strict=True)
            if mode=='fine':assert saved['loaded_sha256']==source_state
            val,_=c.evaluate(model,{target:c.evaluation_pools(data,'val',1024)[target]})
            valerror=None
            if mode!='mixed_zero':
                assert saved['dataset_sha256']==plan['data_sha256'];valerror=max(abs(val['regions'][target][k]-saved['validation']['regions'][target][k]) for k in ['time','mag','total']);assert valerror<5e-5
            metrics,windows=c.evaluate(model,{target:c.evaluation_pools(data,'test',1024)[target]});value=metrics['regions'][target]
            expected=data['regions'][target]['test_end']-data['regions'][target]['val_end'];assert sum(x['events'] for x in windows[target])==expected
            label=f'{target}_{mode}_steps_{updates}_seed_{seed}'
            c.save(REPORTS/f'neural/{label}.json',dict(validation=val['regions'][target],test=value,windows=windows[target],checkpoint=str(folder/'best_model.pth'),checkpoint_sha256=saved['checkpoint_sha256']))
            rows.append(dict(target=target,mode=mode,updates=updates,seed=seed,lr=saved['lr'],best_step=state['step'],events=expected,**value))
            audits.append(dict(target=target,mode=mode,updates=updates,seed=seed,source_validation_replay=error,target_validation_replay=valerror))
            del model;torch.cuda.empty_cache();print('DECAY_DEFAULT_SCORED',label,value,flush=True)
        local=read(selected_dir(target,'local',seed)/'training_complete.json');fine=read(selected_dir(target,'fine',seed)/'training_complete.json')
        assert local['initial_random_sha256']==fine['initial_random_sha256']==src['initial_random_sha256']
        assert local['draw_sha256']==fine['draw_sha256'] and local['exposures']==fine['exposures']
        # Exact target draws must also match the earlier architecture control.
        oldlr=read(BASE/f'configs/{OLD.name}/lr_{target}_mamba2_legacy_local.json')['lr']
        previous=read(OLD/'runs'/f'{target}_mamba2_legacy_local_lr_{oldlr:g}_steps_800_seed_{seed}'/'training_complete.json')
        assert local['draw_sha256']==previous['draw_sha256'] and local['exposures']==previous['exposures']
    import pandas as pd
    frame=pd.DataFrame(rows);frame.to_csv(REPORTS/'neural_metrics.csv',index=False);summary=[]
    for (target,mode,updates),g in frame.groupby(['target','mode','updates'],sort=False):
        summary.append(dict(target=target,mode=mode,updates=int(updates),metrics={k:dict(mean=float(g[k].mean()),std=float(g[k].std(ddof=1))) for k in ['time','mag','total']}))
    c.save(REPORTS/'summary.json',dict(rows=summary));c.save(REPORTS/'final_audit.json',dict(passed=True,source_runs=6,primary_target_scores=18,scored_runs=len(rows),fits=len(list(RUNS.glob('*/training_complete.json'))),checks=audits,paired_initialization_and_exposures=True))
    lines=['# Mamba3默认版：跨区域预训练与川滇/SCEDC迁移','','Mamba3EventTime MIMO rank2、两层64维；仅衰减调制、仅原生相位、屏蔽直接logΔt；legacy时间头、关闭Mixer残差。共同stats-FiLM与离散GR震级头保持原基准。',
           '沿用冻结数据、Mc（川滇3.0/SCEDC2.5）、日期、最早256训练事件标定、1024历史/128评分目标。每地区独立使用截止其验证期前的8区源模型，排除NCEDC重叠。',
           '源训练6400步、目标800步；种子0三学习率分别为本区训练和全量微调选型，种子1/2沿用。源模型用源验证集选检查点，目标模型用目标验证集选检查点。mixed_zero不更新目标权重，但使用训练早期统计校准。',
           '本区与微调的初始化和完整目标抽样一致；新架构与原Mamba2对照也核对了相同目标抽样。预训练额外增加计算。历史测试期为探索性补充。',
           '按验证触发的1600步结果另列；不替换800步主预算。单位nats/事件，越低越好。','','|地区|模式|更新预算|时间NLL|震级NLL|联合NLL|','|---|---|---|---|---|---|']
    for x in summary:
        m=x['metrics'];fmt=lambda k:f"{m[k]['mean']:.5f} ± {m[k]['std']:.5f}"
        lines.append(f"|{x['target']}|{x['mode']}|{x['updates']}|{fmt('time')}|{fmt('mag')}|{fmt('total')}|")
    lines+=['','mixed_zero的6400为源模型训练预算，目标区权重更新为0。逐种子实际选择步数见neural_metrics.csv；完整配置见plan.json。原实现ETAS作为主ETAS对照，旧独立加速版仅作历史补充。']
    (REPORTS/'results_zh.md').write_text('\n'.join(lines)+'\n');print('DECAY_DEFAULT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--finalize',action='store_true')
    p.add_argument('--worker',choices=['source','target','guard']);p.add_argument('--target',choices=TARGETS);p.add_argument('--seed',type=int,default=0);p.add_argument('--mode',choices=['local','fine']);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();configure()
    if a.initialize:initialize()
    elif a.preflight:preflight()
    elif a.run:master(a.workers)
    elif a.finalize:finalize()
    elif a.worker=='guard':control.fit(a.target,FAMILY,a.seed,a.mode,read(CONFIGS/f'lr_{a.target}_{FAMILY}_{a.mode}.json')['lr'],1600)
    elif a.worker:worker(a.worker,a.target,a.seed,800)
