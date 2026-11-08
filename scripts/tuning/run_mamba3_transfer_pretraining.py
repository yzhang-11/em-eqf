"""Paired Mamba3/Oracle/residual regional ablations and past-only transfer sources."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

import mamba3_transfer_common as c

REFERENCE=c.STUDY/'reports/region_capacity_20261105'
REFERENCE_RUNS=c.STUDY/'runs/region_capacity_20261105'
VARIANTS=['mamba3_legacy','mamba3_oracle','mamba3_oracle_residual','mamba2_oracle_residual']
TRANSFER_VARIANTS=['mamba2_legacy','mamba3_oracle_residual']
TRANSFER_DATA=c.STORE/'transfer_sources_before_20110212.pt'


def read(path):return json.loads(Path(path).read_text())


def install_builder():
    import run_multidomain_pretraining as helper
    if getattr(helper.build_model,'regional_ablation',False):return helper.build_model
    original=helper.build_model
    def build(*args,**kwargs):
        model=original(*args,**kwargs)
        if model.inter_time_decoder.parametrization=='oracle':
            # Same unit exponential reference as the legacy head: mean=shape=1.
            # The two parametrizations need different biases to express it.
            with torch.no_grad():model.hypernet_time.bias[:model.num_components].zero_()
        return model
    build.regional_ablation=True;helper.build_model=build
    return build


def load_trainer():
    sys.modules['region_capacity_common']=c
    spec=importlib.util.spec_from_file_location('paired_regional_trainer',c.FROZEN/'scripts/tuning/run_region_capacity_study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def make_backbone(reference,variant):
    result=copy.deepcopy(reference['backbones']['small'])
    result['mixer_time_parametrization']='oracle' if 'oracle' in variant else 'legacy'
    result['mixer_model_config']['use_residual']='residual' in variant
    if variant.startswith('mamba3'):
        result['mixer_model_config']['ssm_cfg']=dict(layer='Mamba3Rotary',d_state=128,headdim=64,
            ngroups=1,expand=2,chunk_size=64,is_mimo=False,rotary_emb_scale_base=None)
    return result


def prepare_transfer_sources():
    if TRANSFER_DATA.exists():return
    original=torch.load(c.DATA,map_location='cpu',weights_only=False)
    train_cut=pd.Timestamp('2009-01-01',tz='UTC').value
    end_cut=pd.Timestamp('2011-02-12',tz='UTC').value
    data=copy.deepcopy(original);counts={}
    for region,spec in data['regions'].items():
        train_end=int(np.searchsorted(spec['physical_ns'],train_cut))
        val_end=int(np.searchsorted(spec['physical_ns'],end_cut))
        assert train_end>256 and val_end>train_end
        spec.update(train_end=train_end,train_used_end=train_end,val_end=val_end,test_end=val_end,
            train_targets=train_end-256,block_starts=list(range(256,train_end,2048)))
        spec['block_targets']=[min(2048,train_end-first) for first in spec['block_starts']]
        # Physically remove all later source observations from the stored input.
        for key in ['ns','physical_ns','magnitude','raw_magnitude','event_id']:spec[key]=spec[key][:val_end]
        assert spec['physical_ns'][-1]<end_cut
        counts[region]=dict(train=train_end-256,validation=val_end-train_end,
            final_timestamp=pd.Timestamp(int(spec['physical_ns'][-1]),tz='UTC').isoformat())
    data['schema']='past_only_transfer_sources_before_20110212_v1'
    data['cutoff_ns']=[train_cut,end_cut,end_cut]
    torch.save(data,TRANSFER_DATA)
    c.save(c.REPORTS/'transfer_source_data.json',dict(path=str(TRANSFER_DATA),sha256=c.sha(TRANSFER_DATA),
        regions=counts,training_before='2009-01-01',validation_end_exclusive='2011-02-12',
        target_training_end='2011-02-12',chuandian_in_pretraining=False,
        rationale='Original 2018 source weights extend beyond the target training cutoff and are not used for transfer.'))


def initialize():
    plan_path=c.CONFIGS/'study_plan.json'
    if plan_path.exists():return read(plan_path)
    c.REPORTS.mkdir(parents=True,exist_ok=True);c.CONFIGS.mkdir(parents=True,exist_ok=True)
    if not (c.REPORTS/'frozen_source.json').exists():c.freeze()
    c.bootstrap();build=install_builder();reference=read(REFERENCE/'summary.json')['plan']
    data=torch.load(c.DATA,map_location='cpu',weights_only=False)
    backbones={v:make_backbone(reference,v) for v in VARIANTS+['mamba2_legacy']}
    parameters={}
    for variant,cfg in backbones.items():
        model=build('stats_film',0,data['vocabulary'],cfg)
        parameters[variant]=sum(p.numel() for p in model.parameters())
        assert len(model.base_model.encoder.layers)==2
        assert model.inter_time_decoder.parametrization==cfg['mixer_time_parametrization']
        assert model.base_model.encoder.use_residual==('residual' in variant)
        del model
    plan=copy.deepcopy(reference)
    plan.update(protocol='mamba3_oracle_residual_paired_20261105_v1',backbones=backbones,
        parameters=parameters,variants=VARIANTS,full_confirmation='mamba3_oracle_residual',
        comparison='All four variants: seed0 nine locals plus two global budgets; combined Mamba3: seeds1,2 repeat.',
        preserved='Same immutable nine-region data, Mc, chronological splits, 2x64 backbone size, target draws, optimizer, budgets and checkpoint rule.',
        required_architecture_difference='Mamba3 SISO has no xPos or convolution; uses actual normalized event times, d_state128, headdim64, expand2.',
        oracle_initialization='Log-mean bias0; shape inverse-softplus(1); legacy rate/shape inverse-softplus(1). Same unit exponential at zero hidden state.',
        reference_mamba2_source='Original capacity-study Mamba2 source, to avoid a later unrelated forward-activation alignment change.',
        transfer_pretraining=dict(variants=TRANSFER_VARIANTS,seeds=[0,1,2],layers=2,width=64,
            source_train_before='2009-01-01',source_validation_end='2011-02-12',global_updates=7200),
        frozen_model_source=str(c.FROZEN),test_selection=False)
    c.save(plan_path,plan);prepare_transfer_sources()
    return plan


def preflight():
    c.bootstrap();build=install_builder();plan=read(c.CONFIGS/'study_plan.json')
    data=torch.load(c.DATA,map_location='cpu',weights_only=False);results={}
    for variant in VARIANTS:
        model=build('stats_film',0,data['vocabulary'],plan['backbones'][variant])
        batch=c.training_batch(data,'japan_east_honshu',1,'long')
        model.train();loss=model.nll_loss(batch,reduction='per_event')['total'].mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        results[variant]=dict(loss=float(loss.detach()),events=int(batch.nll_event_mask.sum()),
            tokens=batch.seq_len,parameters=sum(p.numel() for p in model.parameters()))
        del model;torch.cuda.empty_cache()
    reference_audit=[]
    for seed in [0,1,2]:
        name=f'global_all_regions_small_long_seed_{seed}'
        complete=read(REFERENCE_RUNS/name/'training_complete.json')
        model=build('stats_film',seed,data['vocabulary'],plan['backbones']['mamba2_legacy'])
        assert c.state_hash(model)==complete['initial_state_sha256']
        state=torch.load(REFERENCE_RUNS/name/'best_model.pth',map_location='cuda:0',weights_only=False)
        model.load_state_dict(state['model_state_dict'],strict=True)
        metrics,_=c.evaluate(model,c.evaluation_pools(data,'val',1024))
        error=max(abs(metrics['regions'][r][k]-complete['validation']['regions'][r][k])
                  for r in plan['regions'] for k in ['time','mag','total'])
        assert error<2e-5,(seed,error)
        reference_audit.append(dict(seed=seed,max_absolute_validation_nll_difference=error,
            exact_initial_state_match=True))
        del model;torch.cuda.empty_cache()
    c.save(c.REPORTS/'preflight.json',dict(variants=results,baseline_reuse_audit=reference_audit,passed=True))
    print('PREFLIGHT_COMPLETE',json.dumps(results),flush=True)


def worker(args):
    c.bootstrap();install_builder();plan=read(c.CONFIGS/'study_plan.json')
    if args.kind=='transfer_source':
        c.RUNS=c.RUNS/'transfer_pretraining';c.RUNS.mkdir(exist_ok=True)
        c.CONFIGS=c.CONFIGS/'transfer_pretraining';c.CONFIGS.mkdir(exist_ok=True)
        c.DATA=TRANSFER_DATA;plan['data_sha256']=c.sha(c.DATA)
        plan['protocol']='past_only_transfer_pretraining_20110212_v1'
        c.save(c.CONFIGS/'study_plan.json',plan)
    trainer=load_trainer();data=torch.load(c.DATA,map_location='cpu',weights_only=False)
    assert c.sha(c.DATA)==plan['data_sha256']
    trainer.fit(data,plan,args.variant,'long',args.seed,'global')
    if args.kind=='ablation':
        trainer.fit(data,plan,args.variant,'long',args.seed,'global_compute')
        for region in plan['regions']:trainer.fit(data,plan,args.variant,'long',args.seed,'local',region)


def expected(plan):
    pairs=[(v,0) for v in VARIANTS]+[('mamba3_oracle_residual',s) for s in [1,2]]
    names=[]
    for variant,seed in pairs:
        names.extend(f'{scope}_all_regions_{variant}_long_seed_{seed}' for scope in ['global','global_compute'])
        names.extend(f'local_{r}_{variant}_long_seed_{seed}' for r in plan['regions'])
    return names


def finalize():
    c.bootstrap();build=install_builder();plan=read(c.CONFIGS/'study_plan.json')
    data=torch.load(c.DATA,map_location='cpu',weights_only=False);rows=[]
    pools=c.evaluation_pools(data,'test',1024)
    for name in expected(plan):
        complete=read(c.RUNS/name/'training_complete.json');cfg=complete['config']
        model=build('stats_film',cfg['seed'],data['vocabulary'],cfg['backbone'])
        state=torch.load(c.RUNS/name/'best_model.pth',map_location='cuda:0',weights_only=False)
        model.load_state_dict(state['model_state_dict'],strict=True)
        metrics,windows=c.evaluate(model,{r:pools[r] for r in cfg['sources']})
        c.save(c.RUNS/name/'metrics_context_1024.json',dict(checkpoint=complete['checkpoint'],
            checkpoint_sha256=complete['checkpoint_sha256'],test=metrics,test_windows=windows))
        for region,value in metrics['regions'].items():
            rows.append(dict(variant=cfg['size'],scope=cfg['scope'],seed=cfg['seed'],region=region,
                parameters=complete['parameters'],best_step=complete['best_step'],**value))
        del model;torch.cuda.empty_cache()
    for variant,seed in [(v,0) for v in VARIANTS]+[('mamba3_oracle_residual',s) for s in [1,2]]:
        g=read(c.RUNS/f'global_all_regions_{variant}_long_seed_{seed}/training_complete.json')
        for region in plan['regions']:
            local=read(c.RUNS/f'local_{region}_{variant}_long_seed_{seed}/training_complete.json')
            assert g['initial_state_sha256']==local['initial_state_sha256']
            assert g['training_draw_hashes'][region]==local['training_draw_hashes'][region]
            assert g['scored_event_exposures'][region]==local['scored_event_exposures'][region]
    c.save(c.REPORTS/'neural_results.json',dict(plan=plan,rows=rows,completed_new_fits=len(expected(plan)),
        paired_initialization_and_exposures_verified=True))
    print('ABLATION_TESTS_COMPLETE',flush=True)


def master():
    plan=initialize()
    if not (c.REPORTS/'preflight.json').exists():preflight()
    command=[sys.executable,'-u',str(c.FROZEN/'scripts/tuning/run_mamba3_transfer_pretraining.py'),'--worker']
    # Six past-only source fits are first so downstream work can start while
    # the independent regional ablations continue.
    jobs=[dict(kind='transfer_source',variant=v,seed=s) for s in [0,1,2] for v in TRANSFER_VARIANTS]
    jobs += [dict(kind='ablation',variant=v,seed=0) for v in VARIANTS]
    jobs += [dict(kind='ablation',variant='mamba3_oracle_residual',seed=s) for s in [1,2]]
    active=[];completed=[]
    while jobs or active:
        while jobs and len(active)<4:
            job=jobs.pop(0);label=f"{job['kind']}_{job['variant']}_seed_{job['seed']}"
            log=(c.REPORTS/f'{label}.log').open('a')
            p=subprocess.Popen(command+['--kind',job['kind'],'--variant',job['variant'],'--seed',str(job['seed'])],stdout=log,stderr=subprocess.STDOUT)
            active.append((p,log,job,label));print('START',label,p.pid,flush=True)
        for entry in active[:]:
            p,log,job,label=entry;code=p.poll()
            if code is None:continue
            log.close();active.remove(entry)
            if code:raise RuntimeError(f'{label} failed with code {code}')
            completed.append(job);print('FINISH',label,flush=True)
        if len([j for j in completed if j['kind']=='transfer_source'])==6:
            marker=c.REPORTS/'transfer_sources_complete.json'
            if not marker.exists():c.save(marker,dict(fits=6,variants=TRANSFER_VARIANTS,seeds=[0,1,2],
                data_sha256=c.sha(TRANSFER_DATA),all_source_observations_before='2011-02-12'))
        time.sleep(5)
    finalize();print('PRETRAINING_STUDY_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--initialize',action='store_true')
    p.add_argument('--preflight',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--finalize',action='store_true');p.add_argument('--kind',choices=['ablation','transfer_source'])
    p.add_argument('--variant',choices=VARIANTS+['mamba2_legacy']);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(1)
    if args.run:master()
    elif args.initialize:initialize()
    elif args.preflight:preflight()
    elif args.worker:worker(args)
    elif args.finalize:finalize()
    else:p.error('Choose an action')
