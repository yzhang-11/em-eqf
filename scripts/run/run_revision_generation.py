"""One-GPU queue: full-catalog smoke, frozen Mc sensitivity, then baselines."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from omegaconf import OmegaConf
import pandas as pd

ROOT=Path('/root/autodl-tmp/em_eqf')
OUT=ROOT/'experiments/revision/generation_baselines'
BASE=ROOT/'experiments/revision/main'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    frozen=json.loads((OUT/'frozen_manifest.json').read_text())
    version=digest(OUT/'frozen_manifest.json')
    def verify():
        for row in frozen['source_files']+frozen['configs']+frozen['raw_sources']+frozen['analysis_files']:
            assert digest(Path(row['path']))==row['sha256'],row['path']
    verify()
    save(OUT/'queue_status.json',dict(state='waiting_for_main_calibration',planned_jobs=22))
    while True:
        path=BASE/'reports/calibration/status.json'
        try:
            status=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(1)
            continue
        if status.get('state')=='failed':
            raise RuntimeError('Main calibration failed; follow-up queue held for inspection')
        if status.get('state')=='completed':
            break
        time.sleep(30)
    verify()
    for path in [OUT/'cpu_preflight.json',OUT/'gpu_preflight.json']:
        assert all(row['state']=='passed' for row in json.loads(path.read_text())),path
    results=[]
    def run(job,smoke=False):
        verify()
        name=job['name'];folder=ROOT / 'experiments/revision/archive/smoke/generation_baselines'/name if smoke else Path(job['run_dir'])
        folder.mkdir(parents=True,exist_ok=True)
        config=Path(job['config'])
        assert digest(config)==job['config_sha256'],name
        if smoke:
            cfg=OmegaConf.load(config);cfg.epochs=1;cfg.save_epochs=[]
            config=folder/'smoke_config.yaml'
            if not config.exists():
                OmegaConf.save(cfg,config)
        config_hash=digest(config)
        state_path=folder/'job_status.json'
        if state_path.exists():
            old=json.loads(state_path.read_text())
            assert old['state']=='completed' and old['config_sha256']==config_hash and old['source_manifest_sha256']==version,name
            assert (folder/'metrics_test_best_1.json').exists()
            return old
        state=dict(name=name,group=job['group'],region=job['region'],seed=job['seed'],
            state='running',smoke=smoke,config_sha256=config_hash,source_manifest_sha256=version,
            started_utc=datetime.now(timezone.utc).isoformat())
        save(state_path,state)
        save(OUT/'queue_status.json',dict(state='running',stage='smoke' if smoke else job['group'],job=name))
        start=time.monotonic()
        try:
            cmd=[sys.executable,'-u','main.py','--model',job['model'],'--config',str(config),'--checkpoint_dir',str(folder)]
            for mode in ['train','test']:
                state['phase']=mode;save(state_path,state)
                with (folder/f'{mode}_stdout.log').open('w') as log:
                    process=subprocess.Popen(cmd+['--mode',mode],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                    state['pid']=process.pid;save(state_path,state)
                    code=process.wait()
                if code:
                    raise RuntimeError(f'{mode} exit {code}; inspect {folder}/{mode}_stdout.log')
                if mode=='train':
                    expected=int(OmegaConf.load(config).epochs)
                    assert f'Last model at epoch {expected} saved' in (folder/'run.log').read_text(),name
                    for checkpoint_name in ['best_model_1.pth','last_model_1.pth']:
                        checkpoint=torch.load(folder/checkpoint_name,map_location='cpu',weights_only=False)
                        for key,value in checkpoint['model_state_dict'].items():
                            if torch.is_tensor(value) and value.is_floating_point():
                                assert torch.isfinite(value).all(),(name,key)
                    state['checkpoint_sha256']=digest(folder/'best_model_1.pth')
            metrics=json.loads((folder/'metrics_test_best_1.json').read_text())
            assert metrics and all(torch.isfinite(torch.tensor(v)) for v in metrics.values()),name
            state.update(state='completed',metrics=metrics)
        except Exception as error:
            state.update(state='failed',error=str(error))
            raise
        finally:
            state['elapsed_seconds']=round(time.monotonic()-start,2)
            save(state_path,state)
            print(json.dumps(state),flush=True)
        return state
    smoke=[job for job in plan['baseline_jobs'] if job['seed']==0]
    smoke_results=[run(job,True) for job in smoke]
    save(OUT/'smoke_summary.json',smoke_results)
    for job in plan['mc_sensitivity_jobs']:
        results.append(run(job))
        save(OUT/'followup_results.json',results)
    log=OUT/'logs/mc_magnitude_comparison.log'
    with log.open('w') as file:
        subprocess.run([sys.executable,'-u','scripts/analyze/compare_generation_magnitude.py','--variant','sensitivity'],
            cwd=ROOT,env=ENV,stdout=file,stderr=subprocess.STDOUT,check=True)
    for job in plan['baseline_jobs']:
        results.append(run(job))
        save(OUT/'followup_results.json',results)
    rows=[]
    for region in ['swc','scsn']:
        for model,names,parent in [
            ('EM-EQF',[f'generation_{region}_main_seed_{s}' for s in range(5)],BASE),
            ('RECAST',[f'recast_{region}_main_seed_{s}' for s in range(5)],OUT),
            ('ETAS',[f'etas_{region}_main_seed_0'],OUT),
        ]:
            values=[]
            for name in names:
                metric=json.loads((parent/'runs'/name/'metrics_test_best_1.json').read_text())
                values.append(metric['nll_test_time'])
            rows.append(dict(region=region,model=model,fits=len(values),
                time_NLL_per_day_mean=float(pd.Series(values).mean()),
                time_NLL_per_day_sd=float(pd.Series(values).std(ddof=1)) if len(values)>1 else None,
                inference_scope='fit variation only; no parameter-uncertainty CI; weighted total losses are not compared'))
    pd.DataFrame(rows).to_csv(OUT/'reports/time_nll_summary.csv',index=False)
    save(OUT/'runner_summary.json',dict(completed=len(results),failed=0,smoke_completed=len(smoke_results)))
    save(OUT/'queue_status.json',dict(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),jobs=len(results)))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        save(OUT/'queue_status.json',dict(state='failed',error=str(error),failed_utc=datetime.now(timezone.utc).isoformat()))
        raise
