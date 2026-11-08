"""Factorial MIMO comparison: physical-gap dynamics x external real-time phase."""
import argparse
import copy
import datetime
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import csv

import torch

ROOT = Path(__file__).resolve().parents[2]
ORIGIN = Path('/root/autodl-tmp/em_eqf')
BASE = Path('/root/autodl-fs/em_eqf_training/mamba3_mimo_no_deltat_20261105')
STORE = Path('/root/autodl-fs/em_eqf_training/mamba_time_dynamics_20261105')
REPORTS, RUNS, FROZEN = STORE/'shared/reports', STORE/'runs', STORE/'frozen_project'
DATA = STORE/'dataset_snapshot.pt'
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/run')]
ARMS = [('base_real', 'base', 'real'), ('base_native', 'base', 'native'),
        ('full_dt_native', 'full_dt', 'native'), ('full_dt_real', 'full_dt', 'real'),
        ('decay_only_native', 'decay_only', 'native'), ('decay_only_real', 'decay_only', 'real')]
SEEDS = [0, 1, 2]
CURRENT_UPDATE_MODE = 'base'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class PhysicalGapAdapter:
    """Keep clocks/features intact; add observed elapsed gap / calibration tau."""
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __call__(self, batch):
        result = self.delegate(batch)
        tau = batch.region_params[:, 0, 2:3]
        physical_gap = batch.inter_times / tau
        # Survival/padding entries are not events; keep native DT positive there.
        result['physical_dt'] = torch.where(result['input_mask'].bool(), physical_gap,
                                            torch.ones_like(physical_gap))
        return result


def configure():
    import run_mamba_phase_ablation as phase
    from run_mamba3_mimo_no_deltat import NoDeltaTAdapter
    phase.STORE, phase.REPORTS, phase.RUNS, phase.FROZEN, phase.DATA = STORE, REPORTS, RUNS, FROZEN, DATA
    original_config, original_create = phase.model_config, phase.create_model

    def config(is_mimo):
        assert is_mimo
        result = original_config(True)
        result['mixer_model_config']['ssm_cfg']['time_update_mode'] = CURRENT_UPDATE_MODE
        result['direct_delta_t_input'] = 'masked_zero_after_standardization'
        result['physical_dt_source'] = 'observed_inter_times/calibration_tau; padding=1'
        return result

    def create(is_mimo, phase_mode, seed, vocabulary):
        model = original_create(is_mimo, phase_mode, seed, vocabulary)
        model.base_model.input_adapter = PhysicalGapAdapter(NoDeltaTAdapter(model.base_model.input_adapter))
        return model

    phase.model_config, phase.create_model = config, create
    return phase


def verify_snapshot():
    plan = json.loads((REPORTS/'study_plan.json').read_text())
    assert sha(DATA) == plan['dataset_sha256']
    assert all(sha(FROZEN/name) == h for name,h in plan['source_sha256'].items())
    native = Path(glob.glob('/root/autodl-tmp/venvs/em_eqf-cu128/lib/python*/site-packages/mamba_ssm')[0])
    assert all(sha(native/name) == h for name,h in json.loads((REPORTS/'native_kernels_sha256.json').read_text()).items())
    return plan


def expected_config(phase, arm, phase_mode, seed):
    return dict(arm=arm, is_mimo=True, phase_mode=phase_mode, seed=seed,
        backbone=phase.model_config(True), dataset_sha256=sha(DATA), max_updates=1500,
        checkpoint_selection='validation macro joint NLL', clock_seed_policy='phase-clock-v1')


def baseline(seed):
    return json.loads((REPORTS/f'baseline_seed_{seed}.json').read_text())


def reuse_baseline(phase, seed):
    folder = RUNS/f'base_real_seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    original = baseline(seed)
    source = BASE/'runs'/f'mimo_real_no_deltat_seed_{seed}'/'best_model.pth'
    assert sha(source) == original['checkpoint_sha256']
    if not (folder/'best_model.pth').exists():
        shutil.copyfile(source, folder/'best_model.pth')
    result = copy.deepcopy(original)
    result.pop('test', None)
    result['source_config'] = result['config']
    result['config'] = expected_config(phase, 'base_real', 'real', seed)
    result['reuse'] = dict(source=str(source), source_sha256=sha(source),
        justification='same source checkpoint, coefficients, data, initialization, and masked content features')
    save_json(folder/'config.json', result['config'])
    save_json(folder/'training_complete.json', result)
    print('REUSED_BASELINE', seed, flush=True)
    return result


def preflight(phase, data):
    global CURRENT_UPDATE_MODE
    from run_multidomain_pretraining import make_batch
    train = phase.pools(data, 'train', 0)
    batch = make_batch([train[r][0] for r in sorted(train)[:2]])
    rows = []
    for arm, update_mode, phase_mode in ARMS:
        CURRENT_UPDATE_MODE = update_mode
        model = phase.create_model(True, phase_mode, 0, data['vocabulary'])
        adapter = model.base_model.input_adapter
        actual, original = adapter(batch), adapter.delegate.delegate(batch)
        names = list(adapter.features_input_keys)
        dt_index, mag_index = names.index('log_inter_times'), names.index('mag')
        assert torch.count_nonzero(actual['features'][...,dt_index]) == 0
        assert torch.equal(actual['features'][...,mag_index], original['features'][...,mag_index])
        assert torch.equal(actual['times'], original['times'])
        valid = actual['input_mask'].bool()
        assert (actual['physical_dt'][valid] > 0).all()
        assert torch.isfinite(actual['physical_dt']).all()
        assert torch.all(actual['physical_dt'][~valid] == 1)
        expected_gap = batch.inter_times / batch.region_params[:,0,2:3]
        torch.testing.assert_close(actual['physical_dt'][valid], expected_gap[valid], rtol=0, atol=0)
        initial = phase.tensor_hash(model.state_dict())
        assert initial == baseline(0)['initial_state_sha256']
        loss = model.nll_loss(batch, reduction='per_event')['total'].mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        linear = next(m for m in model.base_model.encoder.input_proj.modules()
                      if isinstance(m,torch.nn.Linear) and m.in_features==2)
        assert torch.count_nonzero(linear.weight.grad[:,dt_index]) == 0
        row = dict(arm=arm, initial_state_sha256=initial, parameters=sum(p.numel() for p in model.parameters()),
            real_phase_clock_preserved=True, magnitude_preserved=True, direct_dt_content_masked=True,
            physical_gaps_positive=True, padding_gap_is_one=True, finite_forward_backward=True,
            feature_order=names, direct_dt_projection_gradient_zero=True, train_loss=float(loss))
        rows.append(row)
        print('PREFLIGHT',json.dumps(row),flush=True)
        del model
        torch.cuda.empty_cache()
    # Reuse is checked on validation only. No new test evaluation before all fits.
    CURRENT_UPDATE_MODE = 'base'
    replay = []
    for seed in SEEDS:
        model = phase.create_model(True,'real',seed,data['vocabulary'])
        state = torch.load(BASE/'runs'/f'mimo_real_no_deltat_seed_{seed}'/'best_model.pth',
                           map_location='cuda:0',weights_only=False)
        model.load_state_dict(state['model_state_dict'],strict=True)
        values,_ = phase.evaluate(model,phase.pools(data,'val',seed))
        old = baseline(seed)['validation']
        error = max(abs(values['regions'][r][k]-old['regions'][r][k])
                    for r in values['regions'] for k in ['time','mag','total'])
        assert error < 1e-6
        replay.append(dict(seed=seed,max_validation_metric_error=error))
        del model,state
        torch.cuda.empty_cache()
    save_json(REPORTS/'preflight.json',dict(rows=rows,reused_baseline_validation_replay=replay))


def seed_worker(seed):
    global CURRENT_UPDATE_MODE
    assert ROOT.resolve() == FROZEN.resolve()
    torch.set_num_threads(1)
    with (RUNS/f'seed_{seed}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_snapshot()
        phase = configure()
        data = torch.load(DATA,map_location='cpu',weights_only=False)
        train,validation = phase.pools(data,'train',seed),phase.pools(data,'val',seed)
        for arm,update_mode,phase_mode in ARMS:
            CURRENT_UPDATE_MODE = update_mode
            result = (reuse_baseline(phase,seed) if arm=='base_real' else
                      phase.fit(data,train,validation,arm,True,phase_mode,seed))
            previous = baseline(seed)
            assert result['initial_state_sha256'] == previous['initial_state_sha256']
            assert result['data_order_sha256'] == previous['data_order_sha256']
            assert result['parameters'] == previous['parameters'] == 158609
        verify_snapshot()
        print('SEED_COMPLETE',seed,flush=True)


def finalize(phase,data):
    global CURRENT_UPDATE_MODE
    completed = {}
    for seed in SEEDS:
        for arm,_,_ in ARMS:
            folder = RUNS/f'{arm}_seed_{seed}'
            r = json.loads((folder/'training_complete.json').read_text())
            assert sha(folder/'best_model.pth') == r['checkpoint_sha256']
            completed[arm,seed] = r
        assert len({completed[a,seed]['initial_state_sha256'] for a,_,_ in ARMS})==1
        assert len({completed[a,seed]['data_order_sha256'] for a,_,_ in ARMS})==1
    rows = []
    for seed in SEEDS:
        tests = phase.pools(data,'test',seed)
        for arm,update_mode,phase_mode in ARMS:
            CURRENT_UPDATE_MODE = update_mode
            folder = RUNS/f'{arm}_seed_{seed}'
            model = phase.create_model(True,phase_mode,seed,data['vocabulary'])
            state = torch.load(folder/'best_model.pth',map_location='cuda:0',weights_only=False)
            model.load_state_dict(state['model_state_dict'],strict=True)
            metrics,windows = phase.evaluate(model,tests)
            if arm=='base_real':
                previous=baseline(seed)['test']
                assert max(abs(metrics['regions'][r][k]-previous['regions'][r][k])
                    for r in metrics['regions'] for k in ['time','mag','total']) < 1e-6
            save_json(folder/'metrics.json',dict(**completed[arm,seed],test=metrics))
            save_json(folder/'test_window_nll.json',windows)
            for region in [*metrics['regions'],'macro']:
                value=metrics['macro'] if region=='macro' else metrics['regions'][region]
                rows.append(dict(arm=arm,update_mode=update_mode,phase_mode=phase_mode,seed=seed,
                    region=region,**value,parameters=158609,best_step=completed[arm,seed]['best_step']))
            print('TEST_COMPLETE',arm,seed,json.dumps(metrics['macro']),flush=True)
            del model,state
            torch.cuda.empty_cache()
    with (REPORTS/'all_results.csv').open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    aggregate=[]
    for arm,update_mode,phase_mode in ARMS:
        for region in ['italy','japan_kyushu','new_zealand','macro']:
            group=[r for r in rows if r['arm']==arm and r['region']==region]
            aggregate.append(dict(arm=arm,update_mode=update_mode,phase_mode=phase_mode,region=region,n=3,
                **{k:dict(mean=statistics.mean(r[k] for r in group),sd=statistics.stdev(r[k] for r in group))
                   for k in ['time','mag','total']}))
    pairs=[('base_real','base_native'),('full_dt_real','full_dt_native'),('decay_only_real','decay_only_native'),
           ('full_dt_native','base_native'),('decay_only_native','base_native'),
           ('full_dt_real','base_real'),('decay_only_real','base_real'),
           ('decay_only_native','full_dt_native'),('decay_only_real','full_dt_real')]
    gains=[]
    for treatment,control in pairs:
        for region in ['italy','japan_kyushu','new_zealand','macro']:
            group={}
            for metric in ['time','mag','total']:
                values=[next(r[metric] for r in rows if r['arm']==control and r['seed']==seed and r['region']==region)
                        -next(r[metric] for r in rows if r['arm']==treatment and r['seed']==seed and r['region']==region)
                        for seed in SEEDS]
                group[metric]=dict(mean=statistics.mean(values),sd=statistics.stdev(values),per_seed=values,
                                   positive_seeds=sum(v>0 for v in values))
            gains.append(dict(treatment=treatment,control=control,region=region,gains=group))
    save_json(REPORTS/'summary.json',dict(completed_arms=18,new_fits=15,reused_fits=3,metrics=aggregate,
        paired_gains=gains,primary_metric='macro time NLL',secondary_metric='macro joint NLL',
        checkpoint_selection='validation macro joint NLL',direct_dt_content_masked=True,parameters=158609,
        all_initialization_and_sampling_matched=True,units='nats/event; lower is better; gain=control-treatment'))


def supervise():
    assert ROOT.resolve()==FROZEN.resolve()
    torch.set_num_threads(1)
    RUNS.mkdir(parents=True,exist_ok=True)
    with (REPORTS/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_snapshot()
        phase=configure()
        data=torch.load(DATA,map_location='cpu',weights_only=False)
        preflight(phase,data)
        children=[]
        for seed in SEEDS:
            log=(REPORTS/f'training_seed_{seed}.log').open('ab',buffering=0)
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--seed',str(seed)],cwd=FROZEN,
                env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1'),stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            children.append(child)
        save_json(REPORTS/'worker_pids.json',{str(s):p.pid for s,p in zip(SEEDS,children)})
        last=0.
        while True:
            status=[p.poll() for p in children]
            if any(code is not None and code!=0 for code in status):
                raise RuntimeError(f'Seed worker failed; inspect individual logs: {status}')
            if all(code==0 for code in status):break
            if time.time()-last>30:
                print('TRAINING_STATUS',json.dumps(dict(completed=len(list(RUNS.glob('*/training_complete.json'))),
                    planned=18,worker_exit_codes=status)),flush=True)
                last=time.time()
            time.sleep(3)
        verify_snapshot()
        finalize(phase,data)
        verify_snapshot()
        save_json(REPORTS/'completed.json',dict(new_fits=15,reused_fits=3,completed_arms=18,snapshots_verified=True,
            completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
        print('TIME_DYNAMICS_COMPLETE',REPORTS/'summary.json',flush=True)


def launch():
    STORE.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    if (REPORTS/'study_plan.json').exists():raise RuntimeError('Existing frozen study; do not replace it')
    old=json.loads((BASE/'reports/study_plan.json').read_text())
    files={}
    for name,h in old['source_sha256'].items():
        source=BASE/'frozen_project'/name
        assert sha(source)==h
        target=FROZEN/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);files[name]=sha(target)
    overrides=['src/models/mamba/mamba3_rotary.py','src/models/mamba/time_update.py',
               'scripts/run/run_mamba_time_dynamics.py','tests/model/test_mamba_time_update.py']
    for name in overrides:
        target=FROZEN/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ORIGIN/name,target);files[name]=sha(target)
    shutil.copyfile(BASE/'dataset_snapshot.pt',DATA)
    for name in ['base_plan.json','environment.json','native_kernels_sha256.json']:
        shutil.copyfile(BASE/'shared/reports'/name,REPORTS/name)
    for seed in SEEDS:
        shutil.copyfile(BASE/'shared/reports'/f'common_initialization_seed_{seed}.pth',REPORTS/f'common_initialization_seed_{seed}.pth')
        shutil.copyfile(BASE/'runs'/f'mimo_real_no_deltat_seed_{seed}'/'metrics.json',REPORTS/f'baseline_seed_{seed}.json')
    references={
        'backbones':ORIGIN/'experiments/revision/mamba_backbones_20261105/reports/summary.json',
        'phase':Path('/root/autodl-fs/em_eqf_training/mamba_phase_ablation_20261105')/'reports/summary.json',
        'no_direct_dt':BASE/'reports/summary.json'}
    for name,source in references.items():shutil.copyfile(source,REPORTS/f'reference_{name}.json')
    plan=dict(protocol='physical_dynamics_by_phase_factorial_v1',
        arms=[dict(arm=a,update_mode=u,phase_mode=p) for a,u,p in ARMS],seeds=SEEDS,
        new_fits=15,reused_fits=3,reused_arm='base_real',updates=1500,validation_every=100,
        input_content='magnitude only; standardized log interval feature is zero, input_dim=2 for matched initialization',
        physical_gap='g=(t_i-t_(i-1))/calibration_tau, positive observed gaps; padding g=1',
        full_dt='DT=learned_DT*g; ADT=A*DT; native phase increments use the effective DT',
        decay_only='DT=learned_DT; ADT=A*DT*g; native phase increments keep the learned DT',
        base='DT=learned_DT; ADT=A*DT',
        real_phase='add fixed-frequency normalized real-time phase; native mode omits only this external term',
        architecture='Mamba3 MIMO rank 2, stats-FiLM, 2 layers, width 64, state 64, head 32',parameters=158609,
        fixed=['all initial weights','training window order','masked direct dt content','architecture','heads',
               'residual enabled','legacy time head','training budget','optimizer and scheduler'],
        split=old['data_split'],train_context=256,train_scored=1024,batch_size=2,evaluation_context=256,evaluation_scored=128,
        optimizer='AdamW',learning_rate=3e-4,weight_decay=.01,gradient_clip=1.,scheduler='10% warmup, cosine floor 5%',
        checkpoint_selection='validation macro joint NLL',primary_metric='test macro time NLL',
        secondary_metrics=['joint NLL','magnitude NLL','per-region NLL'],
        test_access='after all 15 new fits complete; fixed 6-cell matrix and 9 paired contrasts, no test selection',
        execution='one isolated process per seed on the same GPU; no throughput comparison',
        dataset_sha256=old['dataset_sha256'],source_sha256=files,
        created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    save_json(REPORTS/'study_plan.json',plan);verify_snapshot()
    with tarfile.open(REPORTS/'code_snapshot.tar.gz','w:gz') as output:
        for name in files:output.add(FROZEN/name,arcname=name)
    link=ORIGIN/'experiments/revision/mamba_time_dynamics_20261105'
    if not link.exists():link.symlink_to(STORE,target_is_directory=True)
    log=(REPORTS/'training.log').open('ab',buffering=0)
    process=subprocess.Popen([sys.executable,'-u',str(FROZEN/'scripts/run/run_mamba_time_dynamics.py'),'--supervisor'],
        cwd=FROZEN,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1'),
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (REPORTS/'process.pid').write_text(str(process.pid))
    print('LAUNCHED',process.pid,'LOG',REPORTS/'training.log',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--supervisor',action='store_true')
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.run:launch()
    elif args.supervisor:supervise()
    elif args.seed is not None:seed_worker(args.seed)
    else:parser.error('Choose --run, --supervisor or --seed')
