"""Training-log-triggered check: finish warmup before early stopping."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/em_eqf')
STORE=Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105')
BASE=ROOT/'experiments/revision/controls'
ORIGINAL=BASE/'runs/chuandian_multiregion_transfer_20261105'
REPORTS=BASE/'reports/chuandian_warmup_guard_20261105'
CONFIGS=BASE/'configs/chuandian_warmup_guard_20261105'
RUNS=BASE/'runs/chuandian_warmup_guard_20261105'
FROZEN=STORE/'frozen_downstream_warmup_guard'
CORE=FROZEN/'guarded_core.py'
sys.path.insert(0,str(STORE/'frozen_project/scripts/tuning'))
import mamba3_transfer_common as c


def read(path):return json.loads(Path(path).read_text())


def load_core():
    spec=importlib.util.spec_from_file_location('guarded_target_core',CORE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.REPORTS=REPORTS;module.CONFIGS=CONFIGS;module.RUNS=RUNS
    return module


def initialize():
    for path in [REPORTS,CONFIGS,FROZEN,STORE/'downstream_warmup_guard_runs']:path.mkdir(parents=True,exist_ok=True)
    if not RUNS.exists():RUNS.symlink_to(STORE/'downstream_warmup_guard_runs',target_is_directory=True)
    plan_path=CONFIGS/'study_plan.json'
    if plan_path.exists():return read(plan_path)
    original_source=STORE/'frozen_downstream/run_chuandian_encoder_transfer.py'
    text=original_source.read_text();old='if epoch-best_epoch>=10:break'
    assert text.count(old)==1
    CORE.write_text(text.replace(old,'if epoch>=25 and epoch-best_epoch>=10:break'))
    shutil.copy2(Path(__file__),FROZEN/'run_chuandian_warmup_guard.py')
    original_report=BASE/'reports/chuandian_multiregion_transfer_20261105'
    for name in ['data.json','runtime_environment.json','paths.json','dummy_baselines.json']:
        shutil.copy2(original_report/name,REPORTS/name)
    plan=copy.deepcopy(read(BASE/'configs/chuandian_multiregion_transfer_20261105/study_plan.json'))
    rerun=[];reused=[]
    for path in sorted(ORIGINAL.glob('*/training_complete.json')):
        result=read(path);name=path.parent.name
        if 'epochs_completed' in result and result['epochs_completed']<25:
            rerun.append(name)
        else:
            reused.append(name);folder=RUNS/name;folder.mkdir(exist_ok=True)
            checkpoint='best_model.joblib' if '_rf_seed_' in name else 'best_model.pth'
            shutil.copy2(path.parent/checkpoint,folder/checkpoint)
            result['warmup_guard_reused']=True
            result['guard_equivalence']='Original training stopped no earlier than25; changing the minimum to25 leaves its training rule unchanged. RF has no warmup.'
            c.save(folder/'training_complete.json',result)
    plan.update(protocol='chuandian_min25_epoch_warmup_guard_v1',minimum_epochs=25,
        original_protocol='chuandian_two_layer_multiregion_transfer_v1',rerun=rerun,reused=reused,
        selection='Original training duration only: rerun neural fits that stopped before25. No test score used.',
        rationale='15 warmup epochs plus10 patience; prevent ending before full learning rate is exercised.',
        original_source_sha256=c.sha(original_source),guarded_core_sha256=c.sha(CORE),
        sole_training_change='Early stop requires epoch>=25 and epoch-best_epoch>=10.')
    c.save(plan_path,plan);c.save(REPORTS/'plan.json',plan)
    print('GUARD_PLAN',len(rerun),'rerun',len(reused),'reuse',flush=True)
    return plan


def worker(name):
    c.bootstrap();module=load_core();plan=read(CONFIGS/'study_plan.json')
    data=module.torch.load(module.DATA,map_location='cpu',weights_only=False)
    task=data['tasks'];original=read(ORIGINAL/name/'training_complete.json')['config']
    module.fit(data,plan,original['task'],original['family'],original['mode'],original['seed'])


def master():
    plan=initialize();jobs=list(plan['rerun']);active=[]
    command=[sys.executable,'-u',str(FROZEN/'run_chuandian_warmup_guard.py'),'--worker']
    while jobs or active:
        while jobs and len(active)<2:
            name=jobs.pop(0);log=(REPORTS/f'{name}.log').open('a')
            process=subprocess.Popen(command+[name],stdout=log,stderr=subprocess.STDOUT)
            active.append((process,log,name))
        for entry in active[:]:
            process,log,name=entry;code=process.poll()
            if code is None:continue
            log.close();active.remove(entry)
            if code:raise RuntimeError(f'Guarded fit {name} failed with code{code}')
        time.sleep(2)
    module=load_core();module.finalize()
    print('WARMUP_GUARD_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--worker')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.run:master()
    elif args.worker:worker(args.worker)
    else:parser.error('Choose an action')
