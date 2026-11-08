"""Fit the existing project ETAS likelihood and original constrained AdamW setup."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT=Path('/root/autodl-tmp/em_eqf');TAG='original_etas_tpp_20261105'
BASE=ROOT/'experiments/revision/multiregion_pretraining';STORE=Path('/root/autodl-fs/em_eqf_training')/TAG
OLD=STORE.parent/'chuandian_scedc_tpp_20261105'
REPORTS,CONFIGS,RUNS=[BASE/key/TAG for key in ['shared/reports','configs','runs']]
spec=importlib.util.spec_from_file_location('previous_control',OLD/'frozen_runner.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);c=control.c
TARGETS=['chuandian','scedc'];DATA=OLD/'targets.pt'


def read(path):return json.loads(Path(path).read_text())


def setup():
    control.setup()
    # Verify the actual project ETAS and its numerical helpers, rather than
    # silently substituting the independent Gamma-recursion implementation.
    files=['src/models/tpp/etas.py','src/models/tpp/common/etas_utils.py',
           'src/models/tpp/tpp_model.py','src/utils/mask_utils.py','src/train/config_setup.py']
    for name in files:assert c.sha(ROOT/name)==c.sha(control.REFERENCE/name),name


def build(target):
    from src.models.builders import ModelBuilder
    from src.models.tpp.region_mixer_tpp import RegionalGridGR
    cfg=read(CONFIGS/'study_plan.json')['models'][target]

    class ScoredETAS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native=ModelBuilder.by_name('etas')()(OmegaConf.create(cfg),torch.device('cuda:0')).double()
            self.native.b.copy_(torch.tensor(cfg['richter_b_mle'],device='cuda:0',dtype=torch.float64))

        def nll_loss(self,batch,*,reduction='per_event'):
            assert reduction=='per_event'
            time_loss=self.native.nll_loss(batch,reduction='per_event',return_dict=True)['time']
            mark=RegionalGridGR(self.native.b,self.native.M_c,self.native.b.new_tensor(.1))
            mask=batch.nll_event_mask.bool()
            mag_loss=-mark.log_prob(batch.mag,mask=mask).sum(-1)/mask.sum(-1)
            return dict(time=time_loss,mag=mag_loss,total=time_loss+mag_loss)
    return ScoredETAS(),cfg


def initialize():
    setup()
    for p in [STORE,REPORTS,CONFIGS,STORE/'runs']:p.mkdir(parents=True,exist_ok=True)
    if not RUNS.exists():RUNS.symlink_to(STORE/'runs',target_is_directory=True)
    if (CONFIGS/'study_plan.json').exists():
        assert read(CONFIGS/'study_plan.json')['runner_sha256']==c.sha(STORE/'frozen_runner.py');return
    shutil.copy2(Path(__file__),STORE/'frozen_runner.py')
    models={};templates={}
    for target,old_name in [('chuandian','swc'),('scedc','scsn')]:
        p=ROOT/f'experiments/revision/generation_baselines/configs/etas_{old_name}_main_seed_0.yaml'
        cfg=OmegaConf.to_container(OmegaConf.load(p),resolve=True)
        shutil.copy2(p,CONFIGS/f'original_{target}.yaml');templates[str(p)]=c.sha(p)
        old_fit=read(BASE/f'reports/{OLD.name}/etas/{target}_window_classic.json')
        cfg.update(richter_b_mle=old_fit['parameters']['b'],mag_completeness=old_fit['parameters']['mc'],
                   loss_reduction='per_event',etas_grad_checkpoint=False)
        models[target]=cfg
    files=['src/models/tpp/etas.py','src/models/tpp/common/etas_utils.py','src/models/tpp/tpp_model.py',
           'src/utils/mask_utils.py','src/train/config_setup.py','src/train/tpp_train_step.py']
    plan=dict(protocol=TAG,models=models,data_path=str(DATA),data_sha256=c.sha(DATA),
        runner_sha256=c.sha(STORE/'frozen_runner.py'),native_source={name:c.sha(ROOT/name) for name in files},
        original_templates=templates,reference_runner_sha256=c.sha(OLD/'frozen_runner.py'),
        likelihood='Unmodified project src/models/tpp/etas.py, exact pairwise Omori likelihood and original smooth constraints.',
        optimizer='Original AdamW lr=.01,betas=(.9,.99),weight_decay=.005;original get_scheduler(step_warmup);gradient clip3.',
        training='Each epoch accumulates gradients over ALL frozen1024-history+up-to2048-target blocks. Duration-weighted native per_time loss equals full training-window log likelihood per total scored duration.',
        budget='Original500 ChuanDian/3000SCEDC epochs;validation every10/25epochs, including first and last. No test before both fits finish.',
        scoring='Same frozen train/val/test events,Mc,1024history+128target evaluation windows;time+fixed train-MLE0.1-bin GR in nats/event.',
        adaptations=['Frozen benchmark excludes earliest256 calibration events as training targets.','b is train-scored-target discrete GR MLE, as in the shared benchmark.',
                     'Gradient checkpointing disabled for memory-ample GPU;likelihood and gradients unchanged.','Original initializer,optimizer and constraints retained;no artificial seed variability.'])
    c.save(CONFIGS/'study_plan.json',plan);c.save(REPORTS/'plan.json',plan)


def parameters(model):
    n=model.native
    return {**{k:float(getattr(n,k).detach()) for k in ['mu','k','alpha','c','p']},
            'b':float(n.b),'effective_branching_ratio':float(n.effective_branching_ratio())}


def fit(target):
    setup();plan=read(CONFIGS/'study_plan.json');assert c.sha(DATA)==plan['data_sha256']
    model,cfg=build(target);folder=RUNS/target;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_complete.json').exists():return
    data=torch.load(DATA,map_location='cpu',weights_only=False);d=data['regions'][target]
    c._TRAIN_CACHE.clear()
    batches=[c.training_batch(data,target,i,'long') for i in range(len(d['block_starts']))]
    durations=np.array([float((b.t_end-b.t_nll_start).sum()) for b in batches]);weights=durations/durations.sum()
    assert sum(int(b.nll_event_mask.sum()) for b in batches)==d['train_targets']
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=(.9,.99))
    from src.train.config_setup import get_scheduler
    scheduler=get_scheduler(cfg['scheduler_type'],optimizer,OmegaConf.create(cfg),train_dataloader=[None])
    pools={target:c.evaluation_pools(data,'val',1024)[target]}
    history=[];best=float('inf');started=time.time();every=10 if target=='chuandian' else 25
    before=[b.inter_times.clone() for b in batches]
    for epoch in range(1,cfg['epochs']+1):
        model.train();optimizer.zero_grad(set_to_none=True);training=0.
        for batch,weight in zip(batches,weights):
            loss=model.native.nll_loss(batch,reduction='per_time',return_dict=True)['time'].mean()*float(weight)
            assert torch.isfinite(loss);loss.backward();training+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.);assert torch.isfinite(norm)
        optimizer.step();scheduler.step()
        if epoch==1 or epoch%every==0 or epoch==cfg['epochs']:
            validation,_=c.evaluate(model,pools);value=validation['regions'][target]
            assert np.isfinite(value['total'])
            if value['total']<best:
                best=value['total'];torch.save(dict(model_state_dict=model.state_dict(),epoch=epoch,validation=value),folder/'best_model.pth')
            history.append(dict(epoch=epoch,training_per_time=training,validation=value,parameters=parameters(model),seconds=time.time()-started))
            c.save(folder/'progress.json',dict(history=history))
            print('NATIVE_ETAS_PROGRESS',target,epoch,value,parameters(model),round(time.time()-started,1),flush=True)
    assert all(torch.equal(a,b.inter_times) for a,b in zip(before,batches))
    state=torch.load(folder/'best_model.pth',map_location='cpu',weights_only=False)
    c.save(folder/'training_complete.json',dict(target=target,epochs=cfg['epochs'],best_epoch=state['epoch'],
        validation=state['validation'],checkpoint_sha256=c.sha(folder/'best_model.pth'),history=history,
        data_sha256=plan['data_sha256'],train_events=d['train_targets'],raw_clock_unchanged=True,seconds=time.time()-started,test_scored=False))
    print('NATIVE_ETAS_FIT_COMPLETE',target,flush=True)


def direct_window(d,first,count,p):
    left=max(1,first-1024);values=[]
    for i in range(first,first+count):
        past=np.arange(left,i);dt=(d['ns'][i]-d['ns'][i-1])/86400e9
        now=(d['ns'][i]-d['ns'][past])/86400e9;prev=(d['ns'][i-1]-d['ns'][past])/86400e9
        weight=p['k']*10**(p['alpha']*(d['magnitude'][past]-d['metadata'][0]))
        intensity=p['mu']+np.sum(weight*(now+p['c'])**(-p['p']))
        mass=np.exp((1-p['p'])*np.log(prev+p['c']))*(-np.expm1((1-p['p'])*np.log((now+p['c'])/(prev+p['c']))))/(p['p']-1)
        values.append(p['mu']*dt+np.sum(weight*mass)-np.log(intensity))
    return float(np.mean(values))


def finalize():
    setup();plan=read(CONFIGS/'study_plan.json')
    assert all(c.sha(ROOT/name)==value for name,value in plan['native_source'].items())
    assert c.sha(DATA)==plan['data_sha256'] and c.sha(STORE/'frozen_runner.py')==plan['runner_sha256']
    data=torch.load(DATA,map_location='cpu',weights_only=False);rows=[];audits=[]
    for target in TARGETS:
        folder=RUNS/target;complete=read(folder/'training_complete.json');assert c.sha(folder/'best_model.pth')==complete['checkpoint_sha256']
        model,_=build(target);state=torch.load(folder/'best_model.pth',map_location='cuda:0',weights_only=False);model.load_state_dict(state['model_state_dict'],strict=True)
        val,_=c.evaluate(model,{target:c.evaluation_pools(data,'val',1024)[target]})
        error=max(abs(val['regions'][target][k]-complete['validation'][k]) for k in ['time','mag','total']);assert error<1e-8,error
        value,windows=c.evaluate(model,{target:c.evaluation_pools(data,'test',1024)[target]});v=value['regions'][target];d=data['regions'][target]
        assert sum(x['events'] for x in windows[target])==d['test_end']-d['val_end']
        p=parameters(model);assert p['p']>=1.001 and p['effective_branching_ratio']<=.9500001
        errors=[]
        for index in np.unique(np.linspace(0,len(windows[target])-1,6).astype(int)):
            first=d['val_end']+int(index)*128;count=min(128,d['test_end']-first)
            errors.append(abs(direct_window(d,first,count,p)-windows[target][int(index)]['time']))
        assert max(errors)<1e-5,(target,errors)
        rows.append(dict(target=target,best_epoch=state['epoch'],parameters=p,**v))
        audits.append(dict(target=target,validation_replay_error=error,direct_window_max_error=max(errors),raw_clock_unchanged=complete['raw_clock_unchanged']))
        c.save(REPORTS/f'{target}_test.json',dict(metrics=v,parameters=p,windows=windows[target],checkpoint_sha256=complete['checkpoint_sha256']))
        print('NATIVE_ETAS_TEST',rows[-1],flush=True)
    c.save(REPORTS/'summary.json',dict(rows=rows));c.save(REPORTS/'final_audit.json',dict(passed=True,checks=audits,original_source_unchanged=True))
    lines=['# 原项目ETAS：同窗重跑','','直接使用原项目ETAS类及原约束；原AdamW与step_warmup，川滇500/SCEDC3000轮。每轮遍历全部训练窗口并累积梯度。只有两个地区都训练结束才评分测试集。',
      '训练固定1024历史+最多2048目标；测试1024历史+128目标，逐事件条件评分，尾窗口保留。沿用上一轮冻结目录、日期、Mc和训练标定排除规则。',
      '约束：p≥1.001，10000天有效分支比≤0.95，softness0.001；这些是原代码约束，不能把有限期限有效分支比等同于无限期限分支比。',
      '原类只计算时间似然；联合NLL补加固定训练MLE的0.1分箱GR震级项。b与之前同窗基准一致。单位nats/事件，越低越好。',
      '关闭梯度检查点以节省重复计算；原始时钟FP64，模型FP64，原源码字节校验一致。优化器lr=.01、betas=(.9,.99)、weight_decay=.005，梯度裁剪3。',
      '训练归一化仍为原实现per_time；验证选点和最终报告用per_event。验证在同一地区固定区间，归一化分母固定，不改变最优点排序。',
      '历史测试年份已在先前研究中使用，本轮为探索性补充。','','|地区|时间NLL|震级NLL|联合NLL|最佳轮数|','|---|---|---|---|---|']
    for x in rows:lines.append(f"|{x['target']}|{x['time']:.6f}|{x['mag']:.6f}|{x['total']:.6f}|{x['best_epoch']}|")
    lines+=['','完整配置见plan.json；参数与检查见summary.json、final_audit.json；原模板保留在configs，权重在runs。旧加速ETAS结果保留为独立实现对照。']
    (REPORTS/'results_zh.md').write_text('\n'.join(lines)+'\n');print('NATIVE_ETAS_COMPLETE',flush=True)


def master():
    initialize();active=[]
    for target in TARGETS:
        log=(REPORTS/f'{target}.log').open('a')
        p=subprocess.Popen([sys.executable,'-u',str(STORE/'frozen_runner.py'),'--target',target],stdout=log,stderr=subprocess.STDOUT)
        active.append((p,log,target));print('START',target,p.pid,flush=True)
    for p,log,target in active:
        status=p.wait();log.close();assert status==0,(target,status)
    finalize()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--initialize',action='store_true')
    parser.add_argument('--target',choices=TARGETS);parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    if args.initialize:initialize()
    elif args.run:master()
    elif args.finalize:finalize()
    elif args.target:fit(args.target)
