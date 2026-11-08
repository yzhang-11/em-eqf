"""Two-layer encoder transfer on fixed, purged ChuanDian targets."""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import (average_precision_score,roc_auc_score,f1_score,balanced_accuracy_score,
    precision_score,recall_score,brier_score_loss,mean_absolute_error,mean_squared_error,r2_score)

ROOT=Path('/root/autodl-tmp/em_eqf')
FROZEN=Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105/frozen_project')
sys.path.insert(0,str(FROZEN/'scripts/tuning'))
import mamba3_transfer_common as c

BASE=ROOT/'experiments/revision/controls'
CONFIGS=BASE/'configs/chuandian_multiregion_transfer_20261105'
REPORTS=BASE/'reports/chuandian_multiregion_transfer_20261105'
RUNS=BASE/'runs/chuandian_multiregion_transfer_20261105'
DATA=c.STORE/'chuandian_transfer.pt'
MODES=['scratch','frozen','last_block','full']
VARIANTS=['mamba2_legacy','mamba3_oracle_residual']


def read(path):return json.loads(Path(path).read_text())


def source_path(variant,seed):
    return c.RUNS/'transfer_pretraining'/f'global_all_regions_{variant}_long_seed_{seed}'/'best_model.pth'


def parameter_hash(items):
    h=hashlib.sha256()
    for name,value in items:
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize():
    CONFIGS.mkdir(exist_ok=True,parents=True);REPORTS.mkdir(exist_ok=True,parents=True)
    (c.STORE/'downstream_runs').mkdir(exist_ok=True)
    if not RUNS.exists():RUNS.symlink_to(c.STORE/'downstream_runs',target_is_directory=True)
    plan_path=CONFIGS/'study_plan.json'
    if plan_path.exists():return read(plan_path)
    plan=dict(protocol='chuandian_two_layer_multiregion_transfer_v1',tasks=['classification','regression'],
        variants=VARIANTS,modes=MODES,seeds=[0,1,2],layers=2,width=64,
        input_features=['log_inter_times','mag'],input_projection_transferred=True,
        source_training='All source observations before target training cutoff 2011-02-12; source train before2009, validation2009-2011.',
        target_split=read(REPORTS/'data.json')['split_dates'],data_sha256=c.sha(DATA),
        source_encoder='Exact complete base_model.encoder state: input projection, both blocks, final norm. FiLM/TPP heads excluded.',
        adaptation=dict(frozen='Encoder eval and frozen, train attention pool and task head.',
            last_block='Train final encoder block and final norm plus pool/head; earlier encoder stays eval/frozen.',
            full='Train all encoder, pool and head with separate learning rates.',
            scratch='Same architecture, head initialization and learning rates as full; random encoder.'),
        optimizer='AdamW',head_learning_rate=1e-3,encoder_learning_rate=4e-4,weight_decay=.001,
        batch_size=128,max_epochs=150,early_stopping_patience=10,gradient_clip=3.,
        scheduler='10% warmup, step drop0.5 at each20% of maximum epochs, minimum1e-3 of starting LR.',
        classification_loss='focal alpha0.5 gamma2; no sampler or negative filtering',
        regression_loss='smooth_l1 beta0.3 on fixed[3,9] magnitude normalization',
        checkpoint='Classification maximum validation AUPRC; regression minimum validation MAE in magnitude units.',
        classification_threshold='Validation F1 maximum from 0.05..0.95 grid; fixed before test.',
        baselines=['RF on18 existing past-window seismic features plus log elapsed gap',
                   'Two-layer64 LSTM on5 inner windows of18 engineered features',
                   'Two-layer64 LSTM on identical raw normalized event features'],
        baseline_information='All observe the same history and elapsed time since last event; no locations or future inputs.',
        rf_selection='Seed0 validation grid; best setting fixed for seeds0,1,2; no test during tuning.',
        test_selection=False,frozen_model_source=str(FROZEN))
    c.save(plan_path,plan)
    dest=c.STORE/'frozen_downstream';dest.mkdir(exist_ok=True)
    sources=[ROOT/'scripts/tuning/run_chuandian_encoder_transfer.py',ROOT/'scripts/data/prepare_chuandian_encoder_transfer.py']
    manifest={}
    for path in sources:
        shutil.copy2(path,dest/path.name);manifest[path.name]=c.sha(dest/path.name)
    c.save(REPORTS/'downstream_source.json',manifest)
    return plan


class TransferModel(nn.Module):
    def __init__(self,family,mode,backbone,seed):
        super().__init__();torch.manual_seed(seed);self.family=family;self.mode=mode
        if family in VARIANTS:
            from src.models.mamba.mixer_seq import MixerModel
            from src.models.extractors.attn_time_biased_mh import TimeAwareAttnPoolMH
            self.encoder=MixerModel(**backbone['mixer_model_config'],device='cuda',dtype=torch.float32)
            self.pool=TimeAwareAttnPoolMH(64,64,bias_type='log',n_heads=4,agg='concat',
                fuse_mode='concat',use_ln=True,use_var_scale=True,device='cuda')
        else:
            self.encoder=nn.LSTM(18 if family=='lstm_features' else 2,64,num_layers=2,
                batch_first=True,dropout=.3).cuda();self.pool=None
        self.head=nn.Sequential(nn.Linear(65,128),nn.SiLU(),nn.Dropout(.2),nn.Linear(128,1)).cuda()
        self.frozen_names=[]

    def configure(self):
        if self.family not in VARIANTS:return
        for parameter in self.encoder.parameters():parameter.requires_grad_(self.mode in ['scratch','full'])
        if self.mode=='last_block':
            for module in [self.encoder.layers[-1],self.encoder.norm_f]:
                for parameter in module.parameters():parameter.requires_grad_(True)
        self.frozen_names=[name for name,p in self.encoder.named_parameters() if not p.requires_grad]

    def train(self,mode=True):
        super().train(mode)
        if mode and self.family in VARIANTS and self.mode in ['frozen','last_block']:
            self.encoder.eval()
            if self.mode=='last_block':self.encoder.layers[-1].train();self.encoder.norm_f.train()
        return self

    def forward(self,batch):
        if self.family in VARIANTS:
            hidden=self.encoder(features=batch['features'],times=batch['times'],input_mask=batch['mask'])
            pooled=self.pool(hidden,batch['mask'],extra_inputs={'event_time':batch['relative_time']})
        else:
            x=batch['lstm_features'] if self.family=='lstm_features' else batch['features']
            lengths=(torch.full((len(x),),x.shape[1],dtype=torch.long) if self.family=='lstm_features' else batch['lengths'])
            packed=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
            _,(state,_)=self.encoder(packed);pooled=state[-1]
        return self.head(torch.cat([pooled,batch['gap'][:,None]],-1)).squeeze(-1)


def batches(task,indices,seed=None,epoch=0):
    lengths=np.asarray([len(h['times']) for h in task['histories']])
    if seed is None:
        order=np.asarray(indices)[np.argsort(lengths[indices],kind='stable')]
        return [order[i:i+128] for i in range(0,len(order),128)]
    rng=np.random.default_rng(seed*100003+epoch)
    order=rng.permutation(indices);result=[]
    for left in range(0,len(order),512):
        chunk=order[left:left+512];chunk=chunk[np.argsort(lengths[chunk],kind='stable')]
        result.extend(chunk[i:i+128] for i in range(0,len(chunk),128))
    rng.shuffle(result);return result


def collate(data,task,indices):
    histories=[task['histories'][i] for i in indices];lengths=[len(h['times']) for h in histories]
    n=len(indices);length=max(lengths)
    features=np.zeros((n,length,2),dtype=np.float32);times=np.zeros((n,length),dtype=np.float64)
    relative=np.zeros((n,length),dtype=np.float32);mask=np.zeros((n,length),dtype=bool)
    gap=[]
    for row,(hist,size) in enumerate(zip(histories,lengths)):
        features[row,:size]=hist['features'];times[row,:size]=hist['times']
        relative[row,:size]=hist['relative_time'];mask[row,:size]=True
        gap.append(np.log1p(max(0.,1-float(hist['relative_time'][-1]))*task['Twindow']/data['metadata'][2]))
    arrays=dict(features=features,times=times,relative_time=relative,mask=mask,gap=np.asarray(gap,dtype=np.float32),
        lstm_features=task['lstm_features'][indices],labels=task['labels'][indices])
    result={k:torch.from_numpy(v).cuda() for k,v in arrays.items()}
    result['lengths']=torch.tensor(lengths,dtype=torch.long)
    return result


@torch.no_grad()
def predict(model,data,task,indices):
    model.eval();predictions={}
    for ids in batches(task,indices):
        output=model(collate(data,task,ids)).detach().cpu().numpy()
        predictions.update(zip(map(int,ids),map(float,output)))
    return np.asarray([predictions[int(i)] for i in indices])


def metrics(task,y,pred,threshold=None):
    if task=='classification':
        probability=1/(1+np.exp(-np.clip(pred,-50,50)))
        if threshold is None:
            threshold=max(np.arange(.05,.951,.01),key=lambda t:f1_score(y,probability>=t,zero_division=0))
        binary=probability>=threshold
        return dict(auprc=float(average_precision_score(y,probability)),auroc=float(roc_auc_score(y,probability)),
            f1=float(f1_score(y,binary,zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,binary)),
            precision=float(precision_score(y,binary,zero_division=0)),recall=float(recall_score(y,binary,zero_division=0)),
            brier=float(brier_score_loss(y,probability)),threshold=float(threshold))
    actual=3+6*y;prediction=3+6*np.clip(pred,0,1)
    return dict(mae=float(mean_absolute_error(actual,prediction)),rmse=float(np.sqrt(mean_squared_error(actual,prediction))),
        r2=float(r2_score(actual,prediction)),bias=float(np.mean(prediction-actual)))


def objective(task,output,target):
    if task=='regression':return nn.functional.smooth_l1_loss(output,target,beta=.3)
    bce=nn.functional.binary_cross_entropy_with_logits(output,target,reduction='none')
    probability=torch.sigmoid(output);pt=probability*target+(1-probability)*(1-target)
    return (.5*(1-pt).square()*bce).mean()


def preflight():
    c.bootstrap();data=torch.load(DATA,map_location='cpu',weights_only=False)
    backbones=read(c.CONFIGS/'study_plan.json')['backbones'];checks=[]
    for task_name,task in data['tasks'].items():
        lengths=np.asarray([len(h['times']) for h in task['histories']])
        ids=task['splits']['train'][np.argsort(lengths[task['splits']['train']])[-4:]]
        for family,mode in [('mamba2_legacy','last_block'),('mamba3_oracle_residual','frozen'),
                            ('mamba3_oracle_residual','full'),('lstm_raw','scratch'),('lstm_features','scratch')]:
            model=TransferModel(family,mode,backbones.get(family),0)
            if family in VARIANTS:
                payload=torch.load(source_path(family,0),map_location='cpu',weights_only=False)
                prefix='base_model.encoder.'
                state={n[len(prefix):]:v for n,v in payload['model_state_dict'].items() if n.startswith(prefix)}
                model.encoder.load_state_dict(state,strict=True)
            model.configure();model.train()
            before=parameter_hash((n,p) for n,p in model.encoder.named_parameters() if n in model.frozen_names)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
            batch=collate(data,task,ids);output=model(batch);loss=objective(task_name,output,batch['labels'])
            assert torch.isfinite(output).all() and torch.isfinite(loss)
            optimizer.zero_grad();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            after=parameter_hash((n,p) for n,p in model.encoder.named_parameters() if n in model.frozen_names)
            assert before==after
            checks.append(dict(task=task_name,family=family,mode=mode,max_observed_events=int(lengths[ids].max()),
                loss=float(loss.detach()),frozen_parameters_unchanged=True))
            del model,optimizer;torch.cuda.empty_cache()
    c.save(REPORTS/'preflight.json',dict(passed=True,checks=checks,test_used=False))
    print('DOWNSTREAM_PREFLIGHT_COMPLETE',flush=True)


def fit(data,plan,task_name,family,mode,seed):
    name=f'{task_name}_{family}_{mode}_seed_{seed}';folder=RUNS/name;folder.mkdir(exist_ok=True)
    complete=folder/'training_complete.json'
    if complete.exists():return
    backbone=read(c.CONFIGS/'study_plan.json')['backbones'].get(family)
    model=TransferModel(family,mode,backbone,seed)
    head_initial=parameter_hash((n,p) for n,p in model.named_parameters() if not n.startswith('encoder.'))
    initial_encoder=c.state_hash(model.encoder);source=None;loading=None
    if family in VARIANTS and mode!='scratch':
        source=source_path(family,seed)
        state=torch.load(source,map_location='cpu',weights_only=False)
        prefix='base_model.encoder.'
        encoder={k[len(prefix):]:v for k,v in state['model_state_dict'].items() if k.startswith(prefix)}
        assert set(encoder)==set(model.encoder.state_dict())
        model.encoder.load_state_dict(encoder,strict=True)
        loading=dict(tensors=len(encoder),parameters=sum(v.numel() for v in encoder.values()),
            missing_keys=[],unexpected_keys=[],source_sha256=c.sha(source),input_projection_loaded=True)
        assert len(model.encoder.layers)==2
    model.configure();frozen_before=parameter_hash((n,p) for n,p in model.encoder.named_parameters() if n in model.frozen_names)
    source_encoder=c.state_hash(model.encoder)
    encoder_parameters=[p for p in model.encoder.parameters() if p.requires_grad]
    other_parameters=[p for n,p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
    groups=[dict(params=other_parameters,lr=1e-3,initial_lr=1e-3)]
    enc_lr=4e-4 if family in VARIANTS else 1e-3
    if encoder_parameters:groups.append(dict(params=encoder_parameters,lr=enc_lr,initial_lr=enc_lr))
    optimizer=torch.optim.AdamW(groups,weight_decay=.001)
    task=data['tasks'][task_name];train=task['splits']['train'];val=task['splits']['validation']
    cfg=dict(protocol=plan['protocol'],name=name,task=task_name,family=family,mode=mode,seed=seed,
        backbone=backbone,data_sha256=plan['data_sha256'],source_checkpoint=str(source) if source else None,
        source_loading=loading,head_learning_rate=1e-3,encoder_learning_rate=enc_lr,weight_decay=.001,
        max_epochs=150,patience=10,batch_size=128,loss=plan[task_name+'_loss'],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()))
    c.save(CONFIGS/f'{name}.json',cfg)
    history=[];best=math.inf;best_epoch=0;started=time.time()
    for epoch in range(1,151):
        model.train();train_sum=0.;n=0
        factor=min(1.,epoch/15)*max(.001,.5**(epoch//30))
        for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor
        for ids in batches(task,train,seed,epoch):
            batch=collate(data,task,ids);output=model(batch);loss=objective(task_name,output,batch['labels'])
            assert torch.isfinite(loss),(name,epoch)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),3.);assert torch.isfinite(norm)
            optimizer.step();train_sum+=float(loss.detach())*len(ids);n+=len(ids)
        pred=predict(model,data,task,val);value=metrics(task_name,task['labels'][val],pred)
        score=-value['auprc'] if task_name=='classification' else value['mae']
        history.append(dict(epoch=epoch,training_loss=train_sum/n,validation=value,seconds=time.time()-started))
        if score<best:
            best=score;best_epoch=epoch
            torch.save(dict(model_state_dict=model.state_dict(),config=cfg,epoch=epoch,validation=value),folder/'best_model.pth')
        if epoch%5==0 or epoch==1:
            c.save(folder/'progress.json',dict(config=cfg,history=history))
            print(json.dumps(dict(run=name,epoch=epoch,validation=value,seconds=round(time.time()-started,1))),flush=True)
        if epoch-best_epoch>=10:break
    frozen_after=parameter_hash((n,p) for n,p in model.encoder.named_parameters() if n in model.frozen_names)
    assert frozen_before==frozen_after
    state=torch.load(folder/'best_model.pth',map_location='cpu',weights_only=False)
    result=dict(config=cfg,checkpoint=str(folder/'best_model.pth'),checkpoint_sha256=c.sha(folder/'best_model.pth'),
        best_epoch=state['epoch'],validation=state['validation'],epochs_completed=epoch,seconds=time.time()-started,
        head_initial_sha256=head_initial,random_encoder_initial_sha256=initial_encoder,
        loaded_encoder_sha256=source_encoder,frozen_parameters_unchanged=True,history=history)
    c.save(complete,result);print('DOWNSTREAM_TRAINING_COMPLETE',name,flush=True)
    del model,optimizer;torch.cuda.empty_cache()


def rf_features(data,task):
    gap=np.asarray([np.log1p(max(0.,1-float(h['relative_time'][-1]))*task['Twindow']/data['metadata'][2])
                    for h in task['histories']],dtype=np.float32)
    return np.column_stack([task['rf'],gap])


def fit_rf(data,plan):
    import joblib
    for task_name,task in data['tasks'].items():
        x=rf_features(data,task);y=task['labels'];tr=task['splits']['train'];va=task['splits']['validation']
        cls=RandomForestClassifier if task_name=='classification' else RandomForestRegressor
        selected_path=REPORTS/f'rf_selection_{task_name}.json'
        if selected_path.exists():selection=read(selected_path)
        else:
            trials=[]
            for depth,leaf,features in itertools.product([None,8],[1,4,12],['sqrt',1.]):
                cfg=dict(n_estimators=300,max_depth=depth,min_samples_leaf=leaf,max_features=features,
                         random_state=0,n_jobs=4)
                model=cls(**cfg).fit(x[tr],y[tr])
                raw=model.predict_proba(x[va])[:,1] if task_name=='classification' else model.predict(x[va])
                pred=np.log(np.clip(raw,1e-8,1-1e-8)/(1-np.clip(raw,1e-8,1-1e-8))) if task_name=='classification' else raw
                value=metrics(task_name,y[va],pred)
                trials.append(dict(config=cfg,validation=value,score=-value['auprc'] if task_name=='classification' else value['mae']))
            selection=dict(selected=min(trials,key=lambda v:v['score']),trials=trials,test_used=False)
            c.save(selected_path,selection)
        for seed in [0,1,2]:
            name=f'{task_name}_rf_seed_{seed}';folder=RUNS/name;folder.mkdir(exist_ok=True)
            if (folder/'training_complete.json').exists():continue
            cfg=dict(selection['selected']['config'],random_state=seed)
            model=cls(**cfg).fit(x[tr],y[tr]);joblib.dump(model,folder/'best_model.joblib')
            raw=model.predict_proba(x[va])[:,1] if task_name=='classification' else model.predict(x[va])
            pred=np.log(np.clip(raw,1e-8,1-1e-8)/(1-np.clip(raw,1e-8,1-1e-8))) if task_name=='classification' else raw
            result=dict(config=cfg,task=task_name,seed=seed,validation=metrics(task_name,y[va],pred),data_sha256=plan['data_sha256'])
            c.save(folder/'training_complete.json',result)
    print('RF_TRAINING_COMPLETE',flush=True)


def worker(args):
    c.bootstrap();plan=read(CONFIGS/'study_plan.json');assert c.sha(DATA)==plan['data_sha256']
    data=torch.load(DATA,map_location='cpu',weights_only=False)
    if args.rf:fit_rf(data,plan);return
    for task_name in ['classification','regression']:
        for family in VARIANTS:
            for mode in MODES:fit(data,plan,task_name,family,mode,args.seed)
        for family in ['lstm_raw','lstm_features']:fit(data,plan,task_name,family,'scratch',args.seed)


def finalize():
    import joblib
    c.bootstrap();plan=read(CONFIGS/'study_plan.json');data=torch.load(DATA,map_location='cpu',weights_only=False)
    rows=[]
    for complete in sorted(RUNS.glob('*/training_complete.json')):
        result=read(complete);name=complete.parent.name;task_name=name.split('_')[0];task=data['tasks'][task_name]
        ids=task['splits']['test'];y=task['labels'][ids]
        if '_rf_seed_' in name:
            model=joblib.load(complete.parent/'best_model.joblib');x=rf_features(data,task)
            raw=model.predict_proba(x[ids])[:,1] if task_name=='classification' else model.predict(x[ids])
            pred=np.log(np.clip(raw,1e-8,1-1e-8)/(1-np.clip(raw,1e-8,1-1e-8))) if task_name=='classification' else raw
            family='rf';mode='scratch';seed=result['seed']
        else:
            cfg=result['config'];family=cfg['family'];mode=cfg['mode'];seed=cfg['seed']
            model=TransferModel(family,mode,cfg['backbone'],seed)
            state=torch.load(complete.parent/'best_model.pth',map_location='cuda:0',weights_only=False)
            model.load_state_dict(state['model_state_dict'],strict=True)
            pred=predict(model,data,task,ids);del model;torch.cuda.empty_cache()
        threshold=result['validation'].get('threshold')
        value=metrics(task_name,y,pred,threshold)
        np.savez_compressed(complete.parent/'predictions_test.npz',origin=task['origins'][ids],target=y,prediction=pred)
        c.save(complete.parent/'metrics_test.json',dict(metrics=value,forecast_origins_sha256=
            hashlib.sha256(task['origins'][ids].tobytes()).hexdigest(),events=len(ids)))
        rows.append(dict(task=task_name,family=family,mode=mode,seed=seed,n=len(ids),**value))
    assert len(rows)==66,len(rows)
    # Paired task heads and random encoder initializations must match exactly.
    for task in ['classification','regression']:
        for family in VARIANTS:
            for seed in [0,1,2]:
                records=[read(RUNS/f'{task}_{family}_{mode}_seed_{seed}/training_complete.json') for mode in MODES]
                assert len({r['head_initial_sha256'] for r in records})==1
                assert len({r['random_encoder_initial_sha256'] for r in records})==1
                assert all(r['frozen_parameters_unchanged'] for r in records)
    c.save(REPORTS/'test_results.json',dict(plan=plan,rows=rows,
        all_encoder_weights_loaded_strictly=True,frozen_weights_and_paired_heads_verified=True))
    print('CHUANDIAN_TESTS_COMPLETE',flush=True)


def master():
    while not (c.REPORTS/'transfer_sources_complete.json').exists():time.sleep(10)
    plan=initialize()
    if not (REPORTS/'preflight.json').exists():preflight()
    command=[sys.executable,'-u',str(c.STORE/'frozen_downstream/run_chuandian_encoder_transfer.py'),'--worker']
    jobs=[['--rf']]+[['--seed',str(s)] for s in [0,1,2]];active=[]
    while jobs or active:
        # At most two neural workers, alongside the regional study. RF is CPU.
        running_neural=sum('--rf' not in job for _,_,job in active)
        if jobs and (jobs[0]==['--rf'] or running_neural<2):
            job=jobs.pop(0);label='rf' if job==['--rf'] else f'seed_{job[-1]}'
            log=(REPORTS/f'{label}.log').open('a')
            process=subprocess.Popen(command+job,stdout=log,stderr=subprocess.STDOUT)
            active.append((process,log,job));print('START',label,process.pid,flush=True)
        for entry in active[:]:
            process,log,job=entry;code=process.poll()
            if code is None:continue
            active.remove(entry);log.close()
            if code:raise RuntimeError(f'Downstream worker {job} failed: {code}')
        time.sleep(5)
    finalize();print('CHUANDIAN_TRANSFER_STUDY_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--initialize',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--rf',action='store_true')
    p.add_argument('--finalize',action='store_true');p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(1)
    if args.run:master()
    elif args.initialize:initialize()
    elif args.worker:worker(args)
    elif args.finalize:finalize()
    else:p.error('Choose an action')
