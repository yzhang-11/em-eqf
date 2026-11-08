"""Fixed-budget single/global comparison with dense validation and fixed-b controls.

This runner reuses the existing model without changing previous training files.
All selection uses 2014-2015 validation; final tests use 2016-2019 windows.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/run')]
from run_multidomain_pretraining import build_model, load_multidomain_model, make_batch, evaluate, model_settings, save_json
from src.models.tpp.region_mixer_tpp import RegionalGridGR

STUDY=ROOT/'experiments/revision/multiregion_pretraining'
RUNS=STUDY/'runs/multidomain_id/convergence'
CONFIGS=STUDY/'configs/multidomain_id/convergence'
REPORTS=STUDY/'reports/multidomain_id/convergence'
REGIONS=['italy','japan_kyushu','new_zealand']
STEPS=1500
EVERY=100
SEEDS=[0,1,2]
METHOD='stats_film'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def dataset():
    path=ROOT/'data/MultiRegion/catalogs/pilot_2007_2019.pt'
    vocabulary=read(STUDY/'reports/multidomain_id/region_vocabulary.json')
    data=torch.load(path,map_location='cpu',weights_only=False)
    for region,pools in data.items():
        for sequences in pools.values():
            for seq in sequences:
                seq.region_id=torch.full((len(seq),),vocabulary[region],dtype=torch.long)
    return data,vocabulary,sha(path)


def source_files():
    names=['scripts/run/run_multiregion_convergence.py','scripts/run/run_multidomain_pretraining.py',
           'scripts/run/run_multiregion_pretraining.py','src/models/tpp/region_mixer_tpp.py',
           'src/models/tpp/mixer_tpp.py','src/models/tpp/tpp_model.py',
           'src/models/mamba/mamba2_rotary.py','src/models/mha/rotary_embedding_time.py']
    return {p:sha(ROOT/p) for p in names}


def initialize_plan(data_sha):
    plan=dict(protocol='multiregion_convergence_v1',model=METHOD,seeds=SEEDS,regions=REGIONS,
              max_steps=STEPS,validation_every=EVERY,batch_size=16,learning_rate=3e-4,
              optimizer='AdamW',weight_decay=.01,gradient_clip=1.,scheduler='warmup10percent_cosine_floor5percent',
              budget_comparison='same maximum optimizer updates per fitted model; global samples each region uniformly',
              primary_checkpoint='source validation macro NLL; each local uses own region, one global uses all three',
              diagnostic_checkpoint='global checkpoint selected by target-region validation for selection-rule sensitivity',
              magnitude_controls=['dynamic_b','fixed_initial256_b','fixed_train_target_MLE_b'],
              max_fits=12,test_selection=False,dataset_sha256=data_sha,source_sha256=source_files())
    path=CONFIGS/'study_plan.json'
    if path.exists():
        assert read(path)==plan,'Existing convergence plan differs'
    else:
        save_json(path,plan)
    return plan


def state_hash(model):
    digest=hashlib.sha256()
    for key,value in model.state_dict().items():
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit(data,vocabulary,data_sha,seed,sources,name):
    folder=RUNS/name
    cfg=dict(protocol='multiregion_convergence_v1',name=name,method=METHOD,sources=sources,
             targets=sources,seed=seed,optimizer_steps=STEPS,validation_every=EVERY,batch_size=16,
             learning_rate=3e-4,weight_decay=.01,region_id_map=vocabulary,dataset_sha256=data_sha,
             initialization=None,backbone=OmegaConf.to_container(model_settings(METHOD),resolve=True))
    if (folder/'training_complete.json').exists():
        completed=read(folder/'training_complete.json')
        assert completed['config']==cfg
        assert sha(folder/'best_model.pth')==completed['checkpoint_sha256']
        return completed
    folder.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg),CONFIGS/f'{name}.yaml')
    model=build_model(METHOD,seed,vocabulary,cfg['backbone'])
    initial_hash=state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=random.Random(seed)
    best=math.inf
    region_best={r:math.inf for r in sources}
    history=[]
    started=time.time()
    validation={r:data[r]['val'] for r in sources}
    for step in range(1,STEPS+1):
        model.train()
        selected=[rng.choice(data[rng.choice(sources)]['train']) for _ in range(cfg['batch_size'])]
        loss=model.nll_loss(make_batch(selected),reduction='per_event')['total'].mean()
        if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss in {name}')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError(f'Nonfinite gradient in {name}')
        lr=cfg['learning_rate']*min(1.,step/(STEPS//10))*max(.05,.5*(1+math.cos(math.pi*step/STEPS)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        if step%EVERY==0:
            metrics,_=evaluate(model,validation)
            row=dict(step=step,validation=metrics,train_nll=float(loss),elapsed_seconds=time.time()-started)
            history.append(row)
            payload=dict(model_state_dict=model.state_dict(),config=cfg,step=step,validation=metrics)
            if metrics['macro']['total']<best:
                best=metrics['macro']['total'];torch.save(payload,folder/'best_model.pth')
            if len(sources)>1:
                for region in sources:
                    if metrics['regions'][region]['total']<region_best[region]:
                        region_best[region]=metrics['regions'][region]['total']
                        torch.save(payload,folder/f'best_region_{region}.pth')
            save_json(folder/'progress.json',dict(config=cfg,initial_state_sha256=initial_hash,history=history))
            print(json.dumps(dict(run=name,step=step,validation=metrics['macro']['total'],elapsed_seconds=row['elapsed_seconds'])),flush=True)
    checkpoint=folder/'best_model.pth'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    complete=dict(config=cfg,initial_state_sha256=initial_hash,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                  best_step=payload['step'],validation=payload['validation'],elapsed_seconds=time.time()-started)
    save_json(folder/'training_complete.json',complete)
    del model,optimizer,payload
    torch.cuda.empty_cache()
    print(json.dumps(dict(completed=name,best_step=complete['best_step'])),flush=True)
    return complete


def train(seed):
    data,vocabulary,data_sha=dataset()
    initialize_plan(data_sha)
    fit(data,vocabulary,data_sha,seed,REGIONS,f'global_seed_{seed}')
    for region in REGIONS:
        fit(data,vocabulary,data_sha,seed,[region],f'local_{region}_seed_{seed}')


def fixed_b_values(data):
    result={}
    for region in REGIONS:
        sequences=data[region]['train']
        mc,reference,_,_,_,delta,_=sequences[0].region_params[0].tolist()
        indices=torch.cat([((s.mag[64:]-mc)/delta).round() for s in sequences]).numpy()
        mean=float(indices.mean());bins=round((10.-mc)/delta)
        def expected_index(b):
            rate=b*math.log(10)*delta
            return 1/math.expm1(rate)-bins/math.expm1(bins*rate)
        low,high=.2,2.
        for _ in range(80):
            middle=(low+high)/2
            if expected_index(middle)>mean:low=middle
            else:high=middle
        result[region]=dict(initial256=reference,train_mle=(low+high)/2,train_target_events=len(indices),mc=mc,step=delta)
    return result


@torch.no_grad()
def fixed_mark_nll(sequences,b):
    values=[]
    for left in range(0,len(sequences),16):
        batch=make_batch(sequences[left:left+16])
        p=batch.region_params[:,0]
        distribution=RegionalGridGR(torch.full_like(batch.mag,b),p[:,0:1],p[:,5:6])
        log_probability=distribution.log_prob(batch.mag,batch.nll_event_mask.bool())
        values.extend((-(log_probability.sum(-1))/batch.nll_event_mask.sum(-1)).cpu().tolist())
    return dict(mean=float(np.mean(values)),windows=values)


def stats(values):
    values=list(values)
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),n=len(values))


def finalize():
    data,vocabulary,data_sha=dataset()
    plan=initialize_plan(data_sha)
    # All twelve jobs must finish before any test evaluation in this protocol.
    expected=[f'global_seed_{s}' for s in SEEDS]+[f'local_{r}_seed_{s}' for r in REGIONS for s in SEEDS]
    completed={name:read(RUNS/name/'training_complete.json') for name in expected}
    for seed in SEEDS:
        group=[completed[f'global_seed_{seed}']]+[completed[f'local_{r}_seed_{seed}'] for r in REGIONS]
        assert len({r['initial_state_sha256'] for r in group})==1
    baselines=fixed_b_values(data)
    baseline_metrics={r:{split:{kind:fixed_mark_nll(data[r][split],value) for kind,value in [
        ('initial256',baselines[r]['initial256']),('train_mle',baselines[r]['train_mle'])]}
        for split in ['val','test']} for r in REGIONS}
    rows=[]
    for name in expected:
        training=completed[name];cfg=training['config'];folder=RUNS/name
        assert sha(folder/'best_model.pth')==training['checkpoint_sha256']
        model,_=load_multidomain_model(folder/'best_model.pth')
        test,windows=evaluate(model,{r:data[r]['test'] for r in cfg['sources']})
        result=dict(**training,test=test,test_window_nll=windows)
        save_json(folder/'metrics.json',result)
        for region in cfg['sources']:
            metrics=test['regions'][region];validation=training['validation']['regions'][region]
            for mark_mode in ['dynamic','initial256','train_mle']:
                mag=metrics['mag'] if mark_mode=='dynamic' else baseline_metrics[region]['test'][mark_mode]['mean']
                vmag=validation['mag'] if mark_mode=='dynamic' else baseline_metrics[region]['val'][mark_mode]['mean']
                rows.append(dict(kind='global_shared' if name.startswith('global') else 'local',region=region,seed=cfg['seed'],
                                 mark=mark_mode,best_step=training['best_step'],time=metrics['time'],mag=mag,total=metrics['time']+mag,
                                 validation=validation['time']+vmag,checkpoint=str(folder/'best_model.pth')))
        del model
        if name.startswith('global'):
            for region in REGIONS:
                checkpoint=folder/f'best_region_{region}.pth'
                model,config=load_multidomain_model(checkpoint)
                payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                test,_=evaluate(model,{region:data[region]['test']})
                m=test['regions'][region]
                rows.append(dict(kind='global_target_validation',region=region,seed=cfg['seed'],mark='dynamic',best_step=payload['step'],
                                 **m,validation=payload['validation']['regions'][region]['total'],checkpoint=str(checkpoint)))
                del model,payload
        torch.cuda.empty_cache()
    with (REPORTS/'all_results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate=[]
    for kind in ['local','global_shared','global_target_validation']:
        for region in REGIONS:
            for mark in ['dynamic','initial256','train_mle'] if kind!='global_target_validation' else ['dynamic']:
                group=[r for r in rows if r['kind']==kind and r['region']==region and r['mark']==mark]
                assert len(group)==3
                aggregate.append(dict(kind=kind,region=region,mark=mark,**{k:stats(r[k] for r in group) for k in ['validation','time','mag','total']}))
    mark_selection={}
    for kind in ['local','global_shared']:
        means={mark:statistics.mean(r['validation'] for r in rows if r['kind']==kind and r['mark']==mark)
               for mark in ['dynamic','initial256','train_mle']}
        mark_selection[kind]=dict(selected=min(means,key=means.get),validation=means,criterion='mean region/seed validation total; test not used')
    summary=dict(plan=plan,completed_fits=12,results=aggregate,fixed_b=baselines,mark_selection=mark_selection)
    save_json(REPORTS/'summary.json',summary)
    lines=['# 多区训练的长预算对照（2026-11-04）','',
           '同一模型结构、初始化种子、逐区归一化和既有训练/验证/测试切分。每个模型最多 1500 步、batch 16、学习率 3e-4；每 100 步验证并按验证联合 NLL 保存检查点，共 12 次拟合。',
           '单区模型只用本区训练；联合模型每个样本先均匀选地区。等优化步数下，单区看到本区事件的次数约为联合的三倍。三个单区模型总预算是一个联合模型的三倍。',
           '原 1500 步试验每 500 步验证，这次每 100 步验证，可能选择更早的检查点，因此不能把旧结果直接当作相同训练程序。',
           '联合默认只使用一个按三地区验证均值选出的共享检查点；另列按目标区验证选点的敏感性检查，后者需要不同检查点，不能称作一个统一部署模型。',
           '全部 12 次训练完成后才评估测试集；每个种子下四个模型的初始权重摘要一致。NLL 单位 nats/事件，越低越好；± 为三种子样本标准差。','',
           '| 地区 | 单区联合 NLL | 联合共享检查点 NLL | 联合按本区验证选点 NLL |','|---|---:|---:|---:|']
    for region in REGIONS:
        a=[next(x for x in aggregate if x['kind']==k and x['region']==region and x['mark']=='dynamic') for k in ['local','global_shared','global_target_validation']]
        lines.append('| '+region+' | '+' | '.join(f"{x['total']['mean']:.6f} ± {x['total']['sd']:.6f}" for x in a)+' |')
    lines+=['','## 时间和震级分项','',
            '| 地区 | 单区时间 | 联合时间 | 单区震级 | 联合震级 |','|---|---:|---:|---:|---:|']
    for region in REGIONS:
        a=[next(x for x in aggregate if x['kind']==k and x['region']==region and x['mark']=='dynamic') for k in ['local','global_shared']]
        lines.append(f"| {region} | {a[0]['time']['mean']:.6f} | {a[1]['time']['mean']:.6f} | {a[0]['mag']['mean']:.6f} | {a[1]['mag']['mean']:.6f} |")
    lines+=['','## 固定 b 的震级对照','',
            'initial256 使用最早 256 个事件的参考 b；train_mle 仅用本区训练窗中不重复的目标事件，拟合截断离散 GR 的固定 b。它们不使用验证/测试事件估计 b。',
            '这组对照仅替换已有模型的震级输出，时间输出保持同一检查点；不是重新训练一个不带震级损失的模型。验证集决定是否推荐替换，不按测试结果挑选。','',
            '| 地区 | 单区动态 b | 联合动态 b | 固定 initial256 b | 固定训练 MLE b |','|---|---:|---:|---:|---:|']
    for region in REGIONS:
        a=[next(x for x in aggregate if x['kind']==k and x['region']==region and x['mark']=='dynamic') for k in ['local','global_shared']]
        lines.append(f"| {region} | {a[0]['mag']['mean']:.6f} | {a[1]['mag']['mean']:.6f} | {baseline_metrics[region]['test']['initial256']['mean']:.6f} | {baseline_metrics[region]['test']['train_mle']['mean']:.6f} |")
    lines+=['',f'仅按验证集选择的震级输出：单区组 {mark_selection["local"]["selected"]}；联合组 {mark_selection["global_shared"]["selected"]}。',
            '这些仍是三个地区上的研究性对照；1500 步预算不等于已经证明收敛。模型结构和原论文配置未修改。尚未训练全部百万级目录、评估目标区迁移或计数 CRPS/N/M-test。','',
            '配置、训练轨迹、检查点和报告沿用原目录：','',
            str(CONFIGS),str(RUNS),str(REPORTS),'',
            '复现命令：','',
            '```bash','cd /root/autodl-tmp/em_eqf',
            '/root/miniconda3/bin/python scripts/run/run_multiregion_convergence.py --seed 0',
            '/root/miniconda3/bin/python scripts/run/run_multiregion_convergence.py --seed 1',
            '/root/miniconda3/bin/python scripts/run/run_multiregion_convergence.py --seed 2',
            '/root/miniconda3/bin/python scripts/run/run_multiregion_convergence.py --finalize','```','']
    (REPORTS/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(completed=12,mark_selection=mark_selection,report=str(REPORTS/'results.md'))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.finalize:finalize()
    elif args.seed is not None:train(args.seed)
    else:parser.error('Pass --seed or --finalize')


if __name__=='__main__':main()
