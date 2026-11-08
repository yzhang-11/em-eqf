"""Paired single-region / global TPPs, capacity, context, and ETAS controls."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import torch
from region_capacity_common import (ROOT,STUDY,REPORTS,CONFIGS,STORE,RUNS,FROZEN,DATA,SIZES,
    LOCAL_STEPS,SEEDS,sha,save,freeze,bootstrap,backbone,training_batch,evaluation_pools,evaluate,state_hash)


def read(path):return json.loads(Path(path).read_text())


def initialize():
    if (CONFIGS/'study_plan.json').exists():return read(CONFIGS/'study_plan.json')
    if not (REPORTS/'frozen_source.json').exists():freeze()
    bootstrap()
    data=torch.load(DATA,map_location='cpu',weights_only=False)
    from run_multidomain_pretraining import build_model
    configs={s:backbone(s) for s in SIZES};parameters={}
    for size in SIZES:
        model=build_model('stats_film',0,data['vocabulary'],configs[size])
        parameters[size]=sum(p.numel() for p in model.parameters());del model
    torch.cuda.empty_cache()
    plan=dict(protocol='local_global_capacity_20261105_v1',regions=sorted(data['regions']),
        data_sha256=sha(DATA),backbones=configs,parameters=parameters,
        local_updates=LOCAL_STEPS,global_updates=len(data['regions'])*LOCAL_STEPS,
        primary_budget='Global total equals sum of nine local budgets; exact per-region target draws/exposures match.',
        compute_control='Separate global_compute fits: 800 updates with their own 800-step learning-rate schedule, matching each local model.',
        short=dict(history=64,targets_per_window=128,max_batch=16,max_events_per_window=192),
        long=dict(history=1024,max_targets_per_window=2048,batch=1,max_events_per_window=3072),
        paired_training='One region per update; identical target block for short/long and both capacities. Partial tails retained.',
        sampling='Global shuffled balanced region cycles; blocks sampled proportional to target count within region.',
        optimizer='AdamW',learning_rate=3e-4,weight_decay=.01,gradient_clip=1.,
        scheduler='10% warmup, cosine with 5% floor',local_validation_every=100,
        global_primary_validation_every=100*len(data['regions']),
        validation_history=1024,validation_targets=128,event_weighted_within_region=True,
        primary_checkpoint='Global: shared all-region validation macro; local: own validation.',
        screen='Seed 0: two capacities x two lengths x (nine locals + aggregate-budget global + equal-model-budget global).',
        confirm='Select capacity using seed-0 LONG aggregate-budget global validation; seeds 1/2 repeat both LONG global capacities/budgets and nine LONG locals at selected capacity.',
        final_contexts=[64,1024,4096],test_selection=False,
        test_access='Final scoring only after all screen, confirmation, and ETAS fits finish.',
        etas='Per-region full-history normalized temporal ETAS + train-MLE discrete GR; Gamma quadrature validated against direct likelihood.',
        frozen_model_source=str(FROZEN),canonical_model_files_modified=False)
    save(CONFIGS/'study_plan.json',plan)
    return plan


def draw_plan(data,seed,sources,scope):
    draws={};hashes={}
    for region in sources:
        spec=data['regions'][region]
        rng=random.Random(seed*100003+data['vocabulary'][region]*1009)
        draws[region]=rng.choices(range(len(spec['block_starts'])),weights=spec['block_targets'],k=LOCAL_STEPS)
        hashes[region]=hashlib.sha256(json.dumps(draws[region]).encode()).hexdigest()
    if scope=='local':order=[sources[0]]*LOCAL_STEPS
    else:
        rng=random.Random(seed+31337);order=[]
        for _ in range(LOCAL_STEPS):
            cycle=list(sources);rng.shuffle(cycle);order.extend(cycle)
    return draws,order,hashes


def fit(data,plan,size,length,seed,scope,target=None):
    from run_multidomain_pretraining import build_model
    sources=[target] if scope=='local' else plan['regions']
    label=target if target else 'all_regions'
    name=f'{scope}_{label}_{size}_{length}_seed_{seed}';folder=RUNS/name
    cfg=dict(protocol=plan['protocol'],name=name,method='stats_film',size=size,length=length,
        scope=scope,sources=sources,targets=sources,seed=seed,region_id_map=data['vocabulary'],
        backbone=plan['backbones'][size],dataset_sha256=plan['data_sha256'],
        optimizer_steps=plan['global_updates'] if scope=='global' else LOCAL_STEPS,
        learning_rate=plan['learning_rate'],weight_decay=.01,gradient_clip=1.,
        history=64 if length=='short' else 1024,max_targets_per_update=2048)
    done=folder/'training_complete.json'
    if done.exists():
        result=read(done);assert result['config']==cfg and sha(folder/'best_model.pth')==result['checkpoint_sha256']
        print('REUSE',name,flush=True);return result
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(CONFIGS/f'{name}.json',cfg)
        model=build_model('stats_film',seed,data['vocabulary'],cfg['backbone'])
        initial=state_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        draws,order,draw_hashes=draw_plan(data,seed,sources,scope)
        offsets=dict.fromkeys(sources,0);exposures=dict.fromkeys(sources,0)
        validation=evaluation_pools(data,'val',1024);validation={r:validation[r] for r in sources}
        history=[];best=math.inf;equal_best=math.inf;start_step=0
        resume=folder/'last_state.pth'
        if resume.exists():
            state=torch.load(resume,map_location='cuda:0',weights_only=False)
            assert state['config']==cfg
            model.load_state_dict(state['model_state_dict']);opt.load_state_dict(state['optimizer_state'])
            start_step=state['step'];history=state['history'];best=state['best'];equal_best=state['equal_best']
            offsets=state['offsets'];exposures=state['exposures']
            torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
            del state
        started=time.time();torch.cuda.reset_peak_memory_stats()
        maximum=cfg['optimizer_steps']
        for step in range(start_step+1,maximum+1):
            region=order[step-1];block=draws[region][offsets[region]];offsets[region]+=1
            batch=training_batch(data,region,block,length);n=batch.nll_event_mask.sum(-1)
            exposures[region]+=int(n.sum())
            model.train();losses=model.nll_loss(batch,reduction='per_event')
            loss=(losses['total']*n).sum()/n.sum()
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss {name}:{step}')
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError(f'Nonfinite gradient {name}:{step}')
            lr=3e-4*min(1.,step/max(1,maximum//10))*max(.05,.5*(1+math.cos(math.pi*step/maximum)))
            for group in opt.param_groups:group['lr']=lr
            opt.step()
            primary=step%(100*len(sources))==0 if scope=='global' else step%100==0
            equal=(scope=='global' and step<=LOCAL_STEPS and step%100==0)
            if primary or equal:
                metrics,_=evaluate(model,validation)
                row=dict(step=step,train_nll=float(loss.detach()),validation=metrics,
                         seconds=time.time()-started,primary=primary,equal_model_budget=equal)
                history.append(row)
                payload=dict(model_state_dict=model.state_dict(),config=cfg,step=step,validation=metrics)
                if primary and metrics['macro']['total']<best:
                    best=metrics['macro']['total'];torch.save(payload,folder/'best_model.pth')
                if equal and metrics['macro']['total']<equal_best:
                    equal_best=metrics['macro']['total'];torch.save(payload,folder/'equal_model_budget.pth')
                save(folder/'progress.json',dict(config=cfg,history=history,initial_state_sha256=initial))
                torch.save(dict(**payload,optimizer_state=opt.state_dict(),history=history,best=best,
                    equal_best=equal_best,offsets=offsets,exposures=exposures,
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),resume)
                print(json.dumps(dict(run=name,step=step,validation=metrics['macro'],seconds=round(row['seconds'],1))),flush=True)
        payload=torch.load(folder/'best_model.pth',map_location='cpu',weights_only=False)
        result=dict(config=cfg,checkpoint=str(folder/'best_model.pth'),checkpoint_sha256=sha(folder/'best_model.pth'),
            best_step=payload['step'],validation=payload['validation'],initial_state_sha256=initial,
            training_draw_hashes=draw_hashes,scored_event_exposures=exposures,
            completed_region_updates=offsets,parameters=sum(p.numel() for p in model.parameters()),
            elapsed_seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),history=history)
        if scope=='global':result['equal_checkpoint_sha256']=sha(folder/'equal_model_budget.pth')
        save(done,result);print('TRAINING_COMPLETE',name,flush=True)
        del model,opt,payload;torch.cuda.empty_cache()
        return result


def worker(args):
    bootstrap();plan=read(CONFIGS/'study_plan.json');assert sha(DATA)==plan['data_sha256']
    data=torch.load(DATA,map_location='cpu',weights_only=False)
    if args.stage=='screen':
        for length in ['short','long']:
            fit(data,plan,args.size,length,0,'global')
            fit(data,plan,args.size,length,0,'global_compute')
            for region in plan['regions']:fit(data,plan,args.size,length,0,'local',region)
    else:
        chosen=read(REPORTS/'capacity_selection.json')['size']
        for size in SIZES:
            fit(data,plan,size,'long',args.seed,'global')
            fit(data,plan,size,'long',args.seed,'global_compute')
        for region in plan['regions']:fit(data,plan,chosen,'long',args.seed,'local',region)


def select_capacity():
    candidates={size:read(RUNS/f'global_all_regions_{size}_long_seed_0/training_complete.json')['validation']['macro']['total'] for size in SIZES}
    chosen=min(candidates,key=candidates.get)
    save(REPORTS/'capacity_selection.json',dict(size=chosen,validation=candidates,
        criterion='seed-0 LONG global validation macro NLL only; test never used'))
    return chosen


def expected_runs(plan,chosen):
    names=[]
    for size in SIZES:
        for length in ['short','long']:
            names.append(f'global_all_regions_{size}_{length}_seed_0')
            names.append(f'global_compute_all_regions_{size}_{length}_seed_0')
            names.extend(f'local_{r}_{size}_{length}_seed_0' for r in plan['regions'])
    for seed in [1,2]:
        names.extend(f'global_all_regions_{size}_long_seed_{seed}' for size in SIZES)
        names.extend(f'global_compute_all_regions_{size}_long_seed_{seed}' for size in SIZES)
        names.extend(f'local_{r}_{chosen}_long_seed_{seed}' for r in plan['regions'])
    return names


def finalize():
    bootstrap();plan=read(CONFIGS/'study_plan.json');chosen=read(REPORTS/'capacity_selection.json')['size']
    data=torch.load(DATA,map_location='cpu',weights_only=False)
    names=expected_runs(plan,chosen);completed={n:read(RUNS/n/'training_complete.json') for n in names}
    etas=read(REPORTS/'etas/fit_complete.json');assert etas['completed_regions']==len(plan['regions'])
    from run_multidomain_pretraining import build_model
    test_pools={context:evaluation_pools(data,'test',context) for context in [64,1024,4096]}
    rows=[]
    for name,complete in completed.items():
        cfg=complete['config'];model=build_model('stats_film',cfg['seed'],data['vocabulary'],cfg['backbone'])
        contexts=[1024]
        if cfg['length']=='long' and (cfg['scope'].startswith('global') or cfg['size']==chosen):contexts=[64,1024,4096]
        variants=['best_model.pth']
        for filename in variants:
            state=torch.load(RUNS/name/filename,map_location='cuda:0',weights_only=False)
            model.load_state_dict(state['model_state_dict'],strict=True)
            for context in contexts:
                test={r:test_pools[context][r] for r in cfg['sources']}
                metrics,windows=evaluate(model,test)
                save(RUNS/name/f'metrics_{filename[:-4]}_context_{context}.json',dict(config=cfg,
                    checkpoint=str(RUNS/name/filename),checkpoint_sha256=sha(RUNS/name/filename),
                    step=state['step'],validation=state['validation'],test=metrics,test_windows=windows))
                for region,m in metrics['regions'].items():
                    rows.append(dict(scope=cfg['scope'],budget='equal_model_updates' if cfg['scope']=='global_compute' else 'exposure_matched',
                        size=cfg['size'],length=cfg['length'],seed=cfg['seed'],region=region,eval_context=context,
                        **m,parameters=complete['parameters'],best_step=state['step'],checkpoint=str(RUNS/name/filename)))
        del model;torch.cuda.empty_cache()
    for seed in SEEDS:
        g=completed[f'global_all_regions_{chosen}_long_seed_{seed}']
        for region in plan['regions']:
            local=completed[f'local_{region}_{chosen}_long_seed_{seed}']
            assert g['initial_state_sha256']==local['initial_state_sha256']
            assert g['training_draw_hashes'][region]==local['training_draw_hashes'][region]
            assert g['scored_event_exposures'][region]==local['scored_event_exposures'][region]
    with (REPORTS/'neural_results.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(REPORTS/'neural_results.json',dict(plan=plan,selected_capacity=chosen,completed_fits=len(completed),
        paired_initialization_and_exposures_verified=True,rows=rows))
    # ETAS test scoring is likewise delayed until all neural fits are complete.
    subprocess.run(['/root/autodl-tmp/venvs/em_eqf-etas/bin/python',str(ROOT/'scripts/tuning/fit_region_etas.py'),'--finalize'],check=True)
    print('ALL_FITS_AND_TESTS_COMPLETE',REPORTS,flush=True)


def master():
    initialize()
    command=[sys.executable,'-u',str(FROZEN/'scripts/tuning/run_region_capacity_study.py')]
    with (REPORTS/'master.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        baseline_log=(REPORTS/'etas_training.log').open('a')
        baseline=subprocess.Popen(['/root/autodl-tmp/venvs/em_eqf-etas/bin/python','-u',str(ROOT/'scripts/tuning/fit_region_etas.py'),'--fit'],stdout=baseline_log,stderr=subprocess.STDOUT)
        workers=[]
        for size in SIZES:
            log=(REPORTS/f'screen_{size}.log').open('a')
            workers.append((subprocess.Popen(command+['--worker','--stage','screen','--size',size],stdout=log,stderr=subprocess.STDOUT),log))
        for process,log in workers:
            code=process.wait();log.close()
            if code:raise RuntimeError(f'Screen worker failed: {code}')
        chosen=select_capacity();print('CAPACITY_SELECTED',chosen,flush=True)
        workers=[]
        for seed in [1,2]:
            log=(REPORTS/f'confirm_seed_{seed}.log').open('a')
            workers.append((subprocess.Popen(command+['--worker','--stage','confirm','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT),log))
        for process,log in workers:
            code=process.wait();log.close()
            if code:raise RuntimeError(f'Confirmation worker failed: {code}')
        code=baseline.wait();baseline_log.close()
        if code:raise RuntimeError(f'ETAS worker failed: {code}')
        finalize()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--stage',choices=['screen','confirm']);parser.add_argument('--size',choices=list(SIZES))
    parser.add_argument('--seed',type=int,choices=SEEDS,default=0)
    args=parser.parse_args();torch.set_num_threads(1)
    REPORTS.mkdir(parents=True,exist_ok=True);CONFIGS.mkdir(parents=True,exist_ok=True)
    if args.initialize:initialize()
    elif args.run:master()
    elif args.worker:worker(args)
    elif args.finalize:finalize()
    else:parser.error('Choose --run, --initialize, --worker, or --finalize')


if __name__=='__main__':main()
