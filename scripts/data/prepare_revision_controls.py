"""Reviewer controls through existing criterion and selective-load settings."""
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
from src.data.preparation import prepare_data, prepare_data_lstm
from src.models.builders import ModelBuilder
from src.train.config_setup import setup_config, load_model_weights

BASE = ROOT / 'experiments/revision/main'
OUT = ROOT / 'experiments/revision/controls'
SOURCE = BASE / 'runs/pretrain_azdx_main_seed_0/last_model_1.pth'


def plain(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def write_job(name, c, group, variant):
    path = OUT / 'configs' / f'{name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert plain(path) == c, f'Existing control differs: {name}'
    else:
        OmegaConf.save(OmegaConf.create(c), path)
    return dict(name=name, model=c['model'], seed=c['seed'], horizon=c['Tfore'],
                group=group, variant=variant, config=str(path), run_dir=str(OUT / 'runs' / name))


def main():
    logging.basicConfig(level=logging.ERROR)
    assert json.loads((BASE / 'runs/pretrain_azdx_main_seed_0/job_status.json').read_text())['state'] == 'completed'
    jobs = []
    for family in ['reg_azdx', 'reg_scratch', 'lstm']:
        for objective in ['mae', 'mse']:
            for seed in range(5):
                c = plain(BASE / 'configs' / f'{family}_seed_{seed}.yaml')
                c.update(criterion_name=objective, criterion_cfg={})
                name = f'loss_{family}_{objective}_seed_{seed}'
                jobs.append(write_job(name, c, 'loss', objective))

    source_cfg = plain(BASE / 'configs/pretrain_azdx_main_seed_0.yaml')
    for task, template in [('clf', 'clf_azdx_h30_seed_0.yaml'), ('reg', 'reg_azdx_seed_0.yaml')]:
        base = plain(BASE / 'configs' / template)
        base.update(mixer_model_config=copy.deepcopy(source_cfg['mixer_model_config']),
                    learning_rate=.001, encoder_learning_rate=.0004, weight_decay=.001,
                    batch_size=128, accumulation_steps=1, epochs=150,
                    early_stopping_patience=10, early_stopping_min_delta=0.,
                    scheduler_type='step_warmup', warmup_ratio=.1,
                    step_lr_step_size_ratio=.2, step_lr_gamma=.5,
                    use_ema=False, mlp_dropout=.2, mlp_hdw=[128],
                    agg='concat', extractor_cfg={'fuse_mode':'concat','use_ln':True,'use_var_scale':True},
                    freeze_parts=[], exclude_freeze_parts=[], freeze_loaded_only=False,
                    resume_optimizer=False, resume_scheduler=False,
                    encoder_param_keywords=['base_model.encoder'], max_grad_norm=3.)
        for initialization in ['full', 'partial', 'scratch']:
            for seed in range(5):
                c = copy.deepcopy(base);c['seed'] = seed
                if initialization == 'scratch':
                    c.update(resume_path=None, load_specific_parts=None)
                else:
                    parts = ['base_model.encoder'] if initialization == 'full' else ['base_model.encoder.input_proj','base_model.encoder.layers.0']
                    c.update(resume_path=str(SOURCE), load_specific_parts=parts)
                jobs.append(write_job(f'transfer_{task}_{initialization}_seed_{seed}', c, 'transfer', initialization))

    # Prepare each task once, then reuse its args for actual model/load audits.
    contexts = {}
    for task in ['clf', 'reg']:
        path = OUT / 'configs' / f'transfer_{task}_full_seed_0.yaml'
        c = OmegaConf.load(path);c.save_dir=str(OUT / 'reports/preflight' / task)
        _, tr, va, te, ds = prepare_data(c, ROOT / 'data/ChuanDian')
        contexts[task] = c
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)['model_state_dict']
    audits = []
    for task in ['clf', 'reg']:
        names = None
        for variant in ['full','partial','scratch']:
            c = copy.deepcopy(contexts[task])
            original = plain(OUT / 'configs' / f'transfer_{task}_{variant}_seed_0.yaml')
            c.load_specific_parts = original['load_specific_parts']
            torch.manual_seed(0)
            model = ModelBuilder.by_name(c.model)()(c, torch.device('cpu'))
            encoder_names = {k for k in model.state_dict() if k.startswith('base_model.encoder.')}
            loaded = set() if variant == 'scratch' else load_model_weights(model, source, c.load_specific_parts)
            assert loaded <= encoder_names
            for key in loaded:
                assert torch.equal(model.state_dict()[key], source[key])
            if variant == 'full':
                assert loaded == encoder_names
                names = encoder_names
            elif variant == 'partial':
                assert loaded and loaded < names
            audits.append(dict(task=task, variant=variant, encoder_keys=len(encoder_names),
                               loaded_keys=len(loaded), loaded_parameters=sum(model.state_dict()[k].numel() for k in loaded),
                               encoder_parameters=sum(model.state_dict()[k].numel() for k in encoder_names),
                               loaded_names=sorted(loaded)))
    assert audits[0]['loaded_keys'] == audits[3]['loaded_keys']
    assert audits[1]['loaded_keys'] == audits[4]['loaded_keys']

    # Check objective creation through the existing setup entry; no new loss code.
    probe = copy.deepcopy(contexts['reg']);probe.scheduler_type='none'
    for objective, expected in [('mae',torch.nn.L1Loss),('mse',torch.nn.MSELoss)]:
        probe.criterion_name=objective;probe.criterion_cfg={}
        _, criterion, _, _, _ = setup_config(probe, torch.device('cpu'), checkpoint=None, restore_weights=False)
        assert isinstance(criterion, expected)

    # Large-target cutoffs are declared using physics and training labels only.
    c = OmegaConf.load(BASE / 'configs/reg_azdx_seed_0.yaml');c.save_dir=str(OUT/'reports/preflight/target_distribution')
    df,tr,va,te,ds=prepare_data(c,ROOT/'data/ChuanDian')
    y=np.round(np.asarray(ds.inverse_normalize_label(np.asarray(ds.labels))).reshape(-1),3)
    threshold=float(np.quantile(y[tr.dataset.indices],.9))
    distributions=[]
    for label,loader in [('train',tr),('validation',va),('test',te)]:
        idx=np.asarray(loader.dataset.indices)
        for name,value in [('M_ge_6',6.),('training_q90',threshold)]:
            selected=idx[y[idx]>=value-1e-8]
            events=set()
            for i in selected:
                t=ds.forecast_origins[i]
                events.update(df.index[(df.t>=t)&(df.t<t+c.Tfore)&(abs(df.Magnitude-y[i])<1e-6)].tolist())
            distributions.append(dict(split=label,criterion=name,threshold=value,windows=len(idx),
                                       selected_windows=len(selected),distinct_maximum_attaining_events=len(events),
                                       y_min=float(y[idx].min()),y_max=float(y[idx].max())))
    (OUT/'reports/transfer_loading_audit.json').write_text(json.dumps(audits,indent=2))
    pd.DataFrame(distributions).to_csv(OUT/'reports/large_target_support.csv',index=False)
    plan=dict(status='prepared_and_CPU_audited; not yet trained',catalog_protocol_status='frozen',
              parent_protocol=str(BASE/'plan.json'),jobs=jobs,
              fixed_seeds=[0,1,2,3,4],selection='existing minimum validation-loss checkpoint; no test-based selection',
              source_checkpoint=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              loss_baseline='reuse main-regression smooth_l1 runs; beta=0.3 on fixed [3,9] normalized labels',
              transfer_scope='prespecified classification (30 days, Mth=4.5) and regression (180 days); same two-Mamba source architecture and optimizer settings; task-specific history/targets retained')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    print('Control jobs:',len(jobs))
    print(json.dumps([{k:v for k,v in row.items() if k!='loaded_names'} for row in audits]))
    print(pd.DataFrame(distributions).to_string(index=False))


if __name__=='__main__':
    main()
