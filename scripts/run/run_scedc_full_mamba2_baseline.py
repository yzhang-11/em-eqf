"""SCEDC complete-prefix Mamba2Rotary+xPos supplement; matched direct-dt ablations."""
import argparse
import copy
import csv
import datetime
import fcntl
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
ORIGIN=Path('/root/autodl-tmp/em_eqf')
BASE=Path('/root/autodl-fs/em_eqf_training/scedc_full_time_dynamics_20261105')
STORE=Path('/root/autodl-fs/em_eqf_training/scedc_full_mamba2_baseline_20261105')
REPORTS,RUNS,FROZEN=STORE/'reports',STORE/'runs',STORE/'frozen_project'
DATA=STORE/'dataset_snapshot.pt'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/run')]
ARMS=[('mamba2_rotary_xpos_no_dt','base','real',False),
      ('mamba2_rotary_xpos_with_dt','base','real',True)]
SEEDS=[0,1,2]
STEPS=1500
EVERY=100


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class CatalogTimeAdapter:
    def __init__(self,delegate,phase_mode='real',direct_dt=False):
        self.delegate,self.phase_mode,self.direct_dt=delegate,phase_mode,direct_dt
    def __getattr__(self,name):return getattr(self.delegate,name)
    def __call__(self,batch):
        result=self.delegate(batch)
        if not self.direct_dt:
            features=result['features'].clone()
            features[...,list(self.features_input_keys).index('log_inter_times')]=0
            result['features']=features
        if self.phase_mode=='native':result['phase_times']=torch.zeros_like(result['times'])
        tau=self.time_scale_base.to(device=batch.inter_times.device,dtype=batch.inter_times.dtype)
        result['physical_dt']=torch.where(result['input_mask'].bool(),batch.inter_times/tau,
                                          torch.ones_like(batch.inter_times))
        return result


def model_config(data,update_mode,seed):
    template=json.loads((REPORTS/'base_model_template.json').read_text())
    cfg=copy.deepcopy(template['backbones']['mamba2_rotary_xpos'])
    cfg.update(data['stats']);cfg.pop('richter_b',None)
    cfg.update(dataset='SCEDC',use_all_data=False,minibatch_training=False,resume_path=None,
        load_specific_parts=[],mixer_time_parametrization='legacy',loss_reduction='per_event',
        loss_weights=dict(time_weight=1.,mag_weight=1.,b_weight=0.,b_smooth_weight=0.),
        epochs=STEPS,batch_size=1,accumulation_steps=1,learning_rate=3e-4,weight_decay=.01,seed=seed,
        catalog_cfg=dict(root_dir=str(STORE/'catalogs')),event_drop_prob=0.)
    cfg['mixer_model_config']['use_residual']=True
    assert update_mode=='base'
    cfg['mixer_model_config']['ssm_cfg'].update(layer='Mamba2Rotary',use_conv=False,chunk_size=256,rotary_emb_scale_base=8192)
    return cfg


def create_model(data,seed,update_mode,phase_mode,direct_dt):
    from src.models.builders import ModelBuilder
    torch.manual_seed(seed)
    model=ModelBuilder.by_name('mixer_tpp')()(OmegaConf.create(model_config(data,update_mode,seed)),torch.device('cuda:0'))
    model.base_model.input_adapter=CatalogTimeAdapter(model.base_model.input_adapter,phase_mode,direct_dt)
    torch.manual_seed(seed+1000)
    return model


def make_batch(data,split):
    from src.data import Batch,Sequence
    batch=Batch.from_list([Sequence(**data['sequences'][split])]).to(torch.device('cuda:0'))
    spec=data['counts'][split]
    assert batch.seq_len==spec['input_tokens']
    assert int(batch.nll_event_mask.sum())==spec['scored_events']
    assert batch.inter_times.dtype==torch.float64 and batch.arrival_times.dtype==torch.float64
    return batch


@torch.no_grad()
def evaluate(model,batch):
    model.eval()
    raw=model.nll_loss(batch,reduction='none')
    per_event=model.reduce_nll_dict(dict(raw),batch,reduction='per_event')
    per_time=model.reduce_nll_dict(dict(raw),batch,reduction='per_time')
    result=dict(per_event={k:float(v.mean()) for k,v in per_event.items()},
                per_time={k:float(v.mean()) for k,v in per_time.items()})
    assert all(math.isfinite(v) for group in result.values() for v in group.values())
    return result


def verify_snapshot():
    plan=json.loads((REPORTS/'study_plan.json').read_text())
    assert sha(DATA)==plan['dataset_sha256']
    assert all(sha(FROZEN/name)==h for name,h in plan['source_sha256'].items())
    native=Path(glob.glob('/root/autodl-tmp/venvs/em_eqf-cu128/lib/python*/site-packages/mamba_ssm')[0])
    assert all(sha(native/name)==h for name,h in json.loads((REPORTS/'native_kernels_sha256.json').read_text()).items())
    return plan


def preflight(data):
    batch=make_batch(data,'train')
    rows=[];hashes=set()
    for arm,update,phase,direct in ARMS:
        model=create_model(data,0,update,phase,direct)
        adapter=model.base_model.input_adapter
        actual,original=adapter(batch),adapter.delegate(batch)
        names=list(adapter.features_input_keys);dt_idx=names.index('log_inter_times');mag_idx=names.index('mag')
        assert torch.equal(actual['features'][...,mag_idx],original['features'][...,mag_idx])
        assert torch.equal(actual['times'],original['times'])
        assert actual['times'].dtype==torch.float64
        if direct:assert torch.equal(actual['features'],original['features'])
        else:assert torch.count_nonzero(actual['features'][...,dt_idx])==0
        assert (actual['physical_dt']>0).all() and torch.isfinite(actual['physical_dt']).all()
        hashes.add(tensor_hash(model.state_dict()))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        loss=model.nll_loss(batch,reduction='per_event')['total'].mean();loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        linear=next(m for m in model.base_model.encoder.input_proj.modules() if isinstance(m,torch.nn.Linear) and m.in_features==2)
        if direct:assert linear.weight.grad[:,dt_idx].abs().sum()>0
        else:assert torch.count_nonzero(linear.weight.grad[:,dt_idx])==0
        torch.cuda.synchronize()
        row=dict(arm=arm,input_tokens=batch.seq_len,scored_events=int(batch.nll_event_mask.sum()),
            parameters=sum(p.numel() for p in model.parameters()),full_forward_backward=True,
            time_dtype=str(actual['times'].dtype),content_dt=direct,feature_order=names,
            initial_state_sha256=tensor_hash(model.state_dict()),finite_forward_backward=True,
            magnitude_and_clock_preserved=True,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
            seconds=time.perf_counter()-start,train_loss=float(loss.detach()))
        rows.append(row);print('PREFLIGHT',json.dumps(row),flush=True)
        del model,adapter,actual,original,loss
        torch.cuda.empty_cache()
    assert len(hashes)==1
    assert len({r['parameters'] for r in rows})==1
    save_json(REPORTS/'preflight.json',dict(rows=rows,matched_initialization=True))


def fit(data,train,val,seed,spec):
    arm,update,phase,direct=spec
    folder=RUNS/f'{arm}_seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    cfg=dict(protocol='scedc_default_full_prefix_v1',arm=arm,update_mode=update,phase_mode=phase,
        direct_dt_content=direct,seed=seed,max_updates=STEPS,backbone=model_config(data,update,seed),
        dataset_sha256=sha(DATA),input_tokens=train.seq_len,scored_events=int(train.nll_event_mask.sum()),
        checkpoint_selection='validation joint NLL per event',validation_every=EVERY,
        training_input='complete SCEDC train prefix; no windowing or state detachment')
    done=folder/'training_complete.json'
    if done.exists():
        old=json.loads(done.read_text());assert old['config']==cfg
        assert sha(folder/'best_model.pth')==old['checkpoint_sha256'];return old
    save_json(folder/'config.json',cfg)
    model=create_model(data,seed,update,phase,direct)
    initial=tensor_hash(model.state_dict())
    init_path=REPORTS/f'initialization_seed_{seed}.pth'
    if not init_path.exists():torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},init_path)
    else:assert tensor_hash(torch.load(init_path,map_location='cpu',weights_only=False))==initial
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    best=math.inf;history=[];started=time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,STEPS+1):
        model.train()
        out=model.nll_loss(train,reduction='per_event');loss=out['total'].mean()
        if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss {arm}:{seed}:{step}')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError(f'Nonfinite gradient {arm}:{seed}:{step}')
        lr=3e-4*min(1.,step/(STEPS//10))*max(.05,.5*(1+math.cos(math.pi*step/STEPS)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        if step%EVERY==0:
            values=evaluate(model,val)
            row=dict(step=step,train={k:float(v.mean().detach()) for k,v in out.items()},
                validation=values,elapsed_seconds=time.time()-started)
            history.append(row)
            if values['per_event']['total']<best:
                best=values['per_event']['total'];temp=folder/'best_model.tmp'
                torch.save(dict(model_state_dict=model.state_dict(),config=cfg,step=step,validation=values),temp)
                temp.replace(folder/'best_model.pth')
            save_json(folder/'progress.json',dict(config=cfg,history=history))
            print(json.dumps(dict(run=folder.name,step=step,validation=values['per_event'],
                seconds=round(time.time()-started,1))),flush=True)
        del out,loss
    selected=torch.load(folder/'best_model.pth',map_location='cpu',weights_only=False)
    result=dict(config=cfg,initial_state_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),
        best_step=selected['step'],validation=selected['validation'],history=history,
        checkpoint_sha256=sha(folder/'best_model.pth'),elapsed_seconds=time.time()-started,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,scored_event_exposures=STEPS*cfg['scored_events'])
    save_json(done,result);print('TRAINING_COMPLETE',folder.name,flush=True)
    del model,optimizer,selected;torch.cuda.empty_cache()
    return result


def seed_worker(seed):
    assert ROOT.resolve()==FROZEN.resolve();torch.set_num_threads(1)
    with (RUNS/f'seed_{seed}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_snapshot()
        data=torch.load(DATA,map_location='cpu',weights_only=False)
        train,val=make_batch(data,'train'),make_batch(data,'val')
        for spec in ARMS:fit(data,train,val,seed,spec)
        verify_snapshot();print('SEED_COMPLETE',seed,flush=True)


def finalize(data):
    completed={}
    for seed in SEEDS:
        for arm,_,_,_ in ARMS:
            folder=RUNS/f'{arm}_seed_{seed}';r=json.loads((folder/'training_complete.json').read_text())
            assert sha(folder/'best_model.pth')==r['checkpoint_sha256'];completed[arm,seed]=r
        assert len({completed[a,seed]['initial_state_sha256'] for a,_,_,_ in ARMS})==1
        assert len({completed[a,seed]['parameters'] for a,_,_,_ in ARMS})==1
    test=make_batch(data,'test')
    rows=[]
    for seed in SEEDS:
        for arm,update,phase,direct in ARMS:
            folder=RUNS/f'{arm}_seed_{seed}'
            model=create_model(data,seed,update,phase,direct)
            checkpoint=torch.load(folder/'best_model.pth',map_location='cuda:0',weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            values=evaluate(model,test)
            save_json(folder/'metrics.json',dict(**completed[arm,seed],test=values,
                test_input_tokens=test.seq_len,test_scored_events=int(test.nll_event_mask.sum())))
            for reduction,metrics in values.items():
                rows.append(dict(arm=arm,update_mode=update,phase_mode=phase,direct_dt_content=direct,
                    seed=seed,reduction=reduction,**metrics,best_step=checkpoint['step'],
                    parameters=completed[arm,seed]['parameters']))
            print('TEST_COMPLETE',arm,seed,json.dumps(values['per_event']),flush=True)
            del model,checkpoint;torch.cuda.empty_cache()
    with (REPORTS/'all_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate=[]
    for arm,update,phase,direct in ARMS:
        for reduction in ['per_event','per_time']:
            group=[r for r in rows if r['arm']==arm and r['reduction']==reduction]
            aggregate.append(dict(arm=arm,update_mode=update,phase_mode=phase,direct_dt_content=direct,
                reduction=reduction,n=3,parameters=group[0]['parameters'],
                **{k:dict(mean=statistics.mean(r[k] for r in group),sd=statistics.stdev(r[k] for r in group))
                   for k in ['time','mag','total']}))
    pairs=[('mamba2_rotary_xpos_with_dt','mamba2_rotary_xpos_no_dt')]
    gains=[]
    for treatment,control in pairs:
        values={}
        for k in ['time','mag','total']:
            diffs=[next(r[k] for r in rows if r['arm']==control and r['seed']==seed and r['reduction']=='per_event')
                   -next(r[k] for r in rows if r['arm']==treatment and r['seed']==seed and r['reduction']=='per_event')
                   for seed in SEEDS]
            values[k]=dict(mean=statistics.mean(diffs),sd=statistics.stdev(diffs),per_seed=diffs,
                           positive_seeds=sum(x>0 for x in diffs))
        gains.append(dict(treatment=treatment,control=control,gains=values))
    save_json(REPORTS/'summary.json',dict(completed_fits=6,metrics=aggregate,paired_gains=gains,
        counts=data['counts'],stats=data['stats'],input='complete default SCEDC prefixes',
        likelihood='default single-catalog MixerTPP: Weibull mixture time + continuous GR magnitude',
        primary_metric='test time NLL per event',secondary_metric='test joint NLL per event',
        checkpoint_selection='validation joint NLL per event',matched_initialization=True,
        units=dict(per_event='nats/event',per_time='nats/day')))


def supervise():
    assert ROOT.resolve()==FROZEN.resolve();torch.set_num_threads(1)
    RUNS.mkdir(parents=True,exist_ok=True)
    with (REPORTS/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_snapshot()
        data=torch.load(DATA,map_location='cpu',weights_only=False);preflight(data)
        children=[]
        for seed in SEEDS:
            log=(REPORTS/f'training_seed_{seed}.log').open('ab',buffering=0)
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--seed',str(seed)],cwd=FROZEN,
                env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True);children.append(child)
        save_json(REPORTS/'worker_pids.json',{str(s):p.pid for s,p in zip(SEEDS,children)})
        last=0.
        while True:
            status=[p.poll() for p in children]
            if any(code is not None and code!=0 for code in status):raise RuntimeError(f'Seed worker failed: {status}')
            if all(code==0 for code in status):break
            if time.time()-last>30:
                print('TRAINING_STATUS',json.dumps(dict(completed=len(list(RUNS.glob('*/training_complete.json'))),
                    planned=6,worker_exit_codes=status)),flush=True);last=time.time()
            time.sleep(3)
        verify_snapshot();finalize(data);verify_snapshot()
        save_json(REPORTS/'completed.json',dict(completed_fits=6,snapshots_verified=True,
            completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
        print('SCEDC_FULL_COMPLETE',REPORTS/'summary.json',flush=True)


def launch():
    import importlib.metadata
    STORE.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    if not DATA.exists(): shutil.copyfile(BASE/'dataset_snapshot.pt',DATA)
    if (REPORTS/'study_plan.json').exists():raise RuntimeError('Existing frozen study; resume its supervisor')
    assert DATA.is_file()
    data=torch.load(DATA,map_location='cpu',weights_only=False)
    previous=json.loads((BASE/'reports/study_plan.json').read_text());files={}
    for name,h in previous['source_sha256'].items():
        source=BASE/'frozen_project'/name;assert sha(source)==h
        target=FROZEN/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);files[name]=h
    for name in ['scripts/run/run_scedc_full_mamba2_baseline.py']:
        target=FROZEN/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ORIGIN/name,target);files[name]=sha(target)
    shutil.copyfile(BASE/'reports/base_model_template.json',REPORTS/'base_model_template.json')
    shutil.copyfile(BASE/'reports/native_kernels_sha256.json',REPORTS/'native_kernels_sha256.json')
    native=Path(glob.glob('/root/autodl-tmp/venvs/em_eqf-cu128/lib/python*/site-packages/mamba_ssm')[0])
    native_hashes=json.loads((REPORTS/'native_kernels_sha256.json').read_text())
    for file in (native/'ops/triton').glob('ssd*.py'):
        native_hashes[str(file.relative_to(native))]=sha(file)
    native_hashes['modules/mamba2.py']=sha(native/'modules/mamba2.py')
    save_json(REPORTS/'native_kernels_sha256.json',native_hashes)
    plan=dict(protocol='scedc_default_full_prefix_mamba2_supplement_v1',
        arms=[dict(arm=a,update_mode=u,phase_mode=p,direct_dt_content=d) for a,u,p,d in ARMS],
        seeds=SEEDS,max_updates=STEPS,validation_every=EVERY,fresh_fits=6,reused_fits=0,
        catalog='default SCEDC-Standard constructor, local annual snapshot through 2019',
        threshold='M>2.0',split=dict(history_start='1981-01-01',train_score_start='1985-01-01',
            validation_start='2005-01-01',test_start='2014-01-01',end='2020-01-01'),
        full_prefix=True,minibatch_training=False,use_all_data=False,state_detachment=False,batch_size=1,
        counts=data['counts'],stats=data['stats'],normalizer_fit='all training-prefix events only, as default preparation',
        architecture='basic single-catalog MixerTPP, Mamba2Rotary+xPos8192, 2 layers, width/state 64, head 32, no conv',
        heads='legacy Weibull mixture and default continuous Gutenberg-Richter',
        residual=True,dropout=.1,optimizer='AdamW',learning_rate=3e-4,weight_decay=.01,gradient_clip=1.,
        scheduler='10% warmup then cosine with 5% floor',
        loss='time NLL + magnitude NLL; no b smoothing/updater penalty',
        primary_metric='test time NLL per event',secondary_metric='joint NLL per event',
        extra_reporting='per_time nats/day, same full-prefix likelihood',
        checkpoint_selection='validation joint NLL per event',
        test_access='model test likelihood only after all 6 fits; preparation may audit split boundaries and counts',
        fixed=['complete input prefix','score masks','initial weights','dropout RNG sequence','model dimensions',
               'tau/magnitude training normalizers','training budget and schedule'],
        physical_gap='observed current inter_times / adapter fixed tau; invalid padding=1',
        execution='three isolated seed processes; no speed comparison',
        dataset_sha256=sha(DATA),source_sha256=files,
        created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    save_json(REPORTS/'study_plan.json',plan);verify_snapshot()
    with tarfile.open(REPORTS/'code_snapshot.tar.gz','w:gz') as archive:
        for name in files:archive.add(FROZEN/name,arcname=name)
    save_json(REPORTS/'environment.json',dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),packages={k:importlib.metadata.version(k) for k in ['mamba-ssm','triton','tilelang']}))
    link=ORIGIN/'experiments/revision/scedc_full_mamba2_baseline_20261105'
    if not link.exists():link.symlink_to(STORE,target_is_directory=True)
    log=(REPORTS/'training.log').open('ab',buffering=0)
    proc=subprocess.Popen([sys.executable,'-u',str(FROZEN/'scripts/run/run_scedc_full_mamba2_baseline.py'),'--supervisor'],
        cwd=FROZEN,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1'),stdout=log,
        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (REPORTS/'process.pid').write_text(str(proc.pid));print('LAUNCHED',proc.pid,'LOG',REPORTS/'training.log',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--supervisor',action='store_true')
    p.add_argument('--seed',type=int,choices=SEEDS);args=p.parse_args()
    if args.run:launch()
    elif args.supervisor:supervise()
    elif args.seed is not None:seed_worker(args.seed)
    else:p.error('Choose --run, --supervisor or --seed')
