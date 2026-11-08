"""Reuse frozen Mc configurations and the paper's existing baseline models."""
import copy
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.data import Batch
from src.data.preparation import prepare_data_tpp
from src.models.builders import ModelBuilder

BASE=ROOT/'experiments/revision/main'
OUT=ROOT/'experiments/revision/generation_baselines'


def plain(path):
    return OmegaConf.to_container(OmegaConf.load(path),resolve=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    options=parser.parse_args()
    device=torch.device(options.device)
    logging.basicConfig(level=logging.ERROR)
    (OUT/'configs').mkdir(parents=True,exist_ok=True)
    jobs=[]
    templates={
        'recast':ROOT/'checkpoints/rtpp_20260506-173356/config.yaml',
        'etas':ROOT/'checkpoints/etas_20260507-144632/config.yaml',
    }
    for region in ['swc','scsn']:
        main=plain(BASE/'configs'/f'generation_{region}_main_seed_0.yaml')
        for name in ['recast','etas']:
            for seed in (range(5) if name=='recast' else [0]):
                c=plain(templates[name])
                c.update(dataset=main['dataset'],catalog_cfg=copy.deepcopy(main['catalog_cfg']),Mc=main['Mc'],
                    richter_b=main['richter_b'],use_double_precision=True,loss_reduction='per_time',
                    use_all_data=False,use_b_updater=False,batch_size=1,accumulation_steps=1,
                    minibatch_training=False,split_by_time=True,time_order=['train','val','test'],
                    resume_path=None,load_specific_parts=None,save_dir=None,save_epochs=[],
                    seed=seed,epochs=main['epochs'],early_stopping_patience=None,
                    use_ema=False,use_amp=False,bg_model=None,num_workers=0)
                if name=='recast':
                    c['rnn_chunk_len']=4096  # Existing exact chunking; no hidden-state detachment.
                else:
                    c.update(etas_query_chunk_size=512,etas_history_chunk_size=2048,
                             etas_grad_checkpoint=True,etas_enforce_subcritical=True,
                             etas_enforce_p_gt_one=True)
                job_name=f'{name}_{region}_main_seed_{seed}'
                path=OUT/'configs'/f'{job_name}.yaml'
                if path.exists():
                    previous=plain(path)
                    if previous!=c:
                        expected=copy.deepcopy(c);expected['early_stopping_patience']=0
                        assert previous==expected and not (OUT/'runs'/job_name/'job_status.json').exists(),job_name
                        OmegaConf.save(OmegaConf.create(c),path)
                else:
                    OmegaConf.save(OmegaConf.create(c),path)
                jobs.append(dict(name=job_name,group=name,region=region,seed=seed,model=c['model'],
                    config=str(path),config_sha256=digest(path),run_dir=str(OUT/'runs'/job_name)))
    sensitivity=[]
    for seed in range(5):
        for region in ['swc','scsn']:
            name=f'generation_{region}_sensitivity_seed_{seed}'
            path=BASE/'configs'/f'{name}.yaml'
            c=plain(path);main=plain(BASE/'configs'/f'generation_{region}_main_seed_{seed}.yaml')
            changed={key for key in set(c)|set(main) if c.get(key)!=main.get(key)}
            assert changed=={'Mc','catalog_cfg','richter_b'},changed
            sensitivity.append(dict(name=name,group='mc_sensitivity',region=region,seed=seed,
                model=c['model'],config=str(path),config_sha256=digest(path),run_dir=str(BASE/'runs'/name)))
    plan=dict(state='prepared; CPU preflight recorded separately',fixed_seeds=list(range(5)),
        selection='minimum validation loss checkpoint; fixed epochs; no test-based model or seed selection',
        budget='same maximum epochs per catalog as EM-EQF: SWC 500, SCSN 3000; no new hyperparameter search',
        etas_variant='constant-background subcritical ETAS with p>1; common existing SCSN constraint settings for both catalogs',
        recast_variant='existing rtpp GRU-32 and 32-component Weibull time density; fixed train-fitted GR magnitude law',
        etas_replicates='one deterministic fixed-initialization optimization per catalog; no artificial seed variability claimed',
        baseline_templates={k:str(v) for k,v in templates.items()},
        caveat='Different Mc defines different target catalogs; do not rank absolute NLL across cutoffs',
        matched_information='same immutable source, Mc, exact train/validation/test intervals and available history as EM-EQF',
        baseline_jobs=jobs,mc_sensitivity_jobs=sensitivity)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    audit=[]
    for region in ['swc','scsn']:
        for name in ['recast','etas']:
            path=OUT/'configs'/f'{name}_{region}_main_seed_0.yaml'
            c=OmegaConf.load(path)
            _,tr,va,te,cat=prepare_data_tpp(c,ROOT/'data'/c.dataset)
            seq=cat.train[0]
            times=seq.arrival_times.detach().cpu().numpy()
            index=int(np.searchsorted(times,seq.t_nll_start))
            start=float(times[max(0,index-128)])-1e-8
            end=float(times[index+64])
            short=seq.get_subsequence(max(seq.t_start,start),end).double()
            batch=Batch.from_list([short]).to(device)
            before=batch.inter_times.clone()
            model=ModelBuilder.by_name(c.model)()(c,device)
            row=dict(name=name,region=region,clock_dtype=str(batch.inter_times.dtype),
                parameter_dtype=str(next(model.parameters()).dtype),train_events=c.num_events_train,
                validation_events=c.num_events_val,test_events=c.num_events_test,richter_b=c.richter_b_mle)
            try:
                out=model.nll_loss(batch,return_dict=True,reduction='per_time')
                loss=out['total'].mean();assert torch.isfinite(loss)
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                torch.testing.assert_close(batch.inter_times,before,rtol=0,atol=0)
                row.update(state='passed',small_window_loss=float(loss.detach()),raw_clock_unchanged=True)
                if name=='recast' and options.device=='cuda':
                    model.eval()
                    past=short.get_subsequence(short.t_start,float(short.arrival_times[-2])).to(device)
                    past.t_nll_start=past.t_start
                    sample=model.sample(batch_size=2,duration=2.,past_seq=past,return_sequences=True)
                    for sequence in sample:
                        assert sequence.inter_times.dtype==torch.float64
                        assert torch.isfinite(sequence.inter_times).all() and (sequence.inter_times>=0).all()
                        assert torch.isfinite(sequence.mag).all()
                    row['two_day_forecast_counts']=[len(s) for s in sample]
            except Exception as error:
                row.update(state='failed',error=repr(error))
            audit.append(row)
            print(row,flush=True)
    (OUT/('gpu_preflight.json' if options.device=='cuda' else 'cpu_preflight.json')).write_text(json.dumps(audit,indent=2))
    print('Prepared baseline jobs',len(jobs),'existing sensitivity jobs',len(sensitivity),flush=True)


if __name__=='__main__':
    main()
