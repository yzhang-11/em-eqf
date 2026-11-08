"""Validate the Blackwell environment with real GPU forward/backward kernels."""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
REPORTS=ROOT/'experiments/revision/multiregion_pretraining/reports'
REPORTS.mkdir(parents=True,exist_ok=True)
SCRATCH=Path('/root/autodl-tmp/tmp/em_eqf-sm120-cache')
SCRATCH.mkdir(parents=True,exist_ok=True)
for key,suffix in [('TRITON_CACHE_DIR','triton'),('TORCH_EXTENSIONS_DIR','torch'),('TILELANG_CACHE_DIR','tilelang')]:
    os.environ.setdefault(key,str(SCRATCH/suffix))
try:
    distribution=importlib.metadata.distribution('nvidia-cuda-nvcc')
    nvcc=next(Path(distribution.locate_file(p)) for p in distribution.files if str(p).endswith('/bin/nvcc'))
    os.environ['CUDA_HOME']=str(nvcc.parent.parent)
    os.environ['PATH']=str(nvcc.parent)+os.pathsep+os.environ.get('PATH','')
except (importlib.metadata.PackageNotFoundError,StopIteration):
    pass
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/run')]

import torch


def exercise(name,model,x):
    print('BEGIN',name,tuple(x.shape),flush=True)
    torch.cuda.reset_peak_memory_stats()
    started=time.time()
    y=model(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    loss=y.float().square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    torch.optim.AdamW(model.parameters(),lr=1e-4).step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    torch.cuda.synchronize()
    result=dict(name=name,shape=list(x.shape),loss=float(loss),gradients_finite=True,optimizer_step_passed=True,
                elapsed_seconds=time.time()-started,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    result['configuration']={key:getattr(model,key,None) for key in ['d_state','headdim','chunk_size','is_mimo','mimo_rank']}
    print(json.dumps(result),flush=True)
    return result


def project_check(events,history):
    import numpy as np
    import pandas as pd
    from src.data import Sequence
    from src.models.tpp.region_mixer_tpp import attach_region
    from run_multidomain_pretraining import build_model,make_batch
    description=json.loads((REPORTS/'pilot_data_20261104.json').read_text())
    data=torch.load(ROOT/'data/MultiRegion/catalogs/pilot_2007_2019.pt',weights_only=False)
    sequences=[]
    vocabulary={'<unknown>':0,'italy':1,'japan_kyushu':2,'new_zealand':3}
    for region in ['italy','japan_kyushu','new_zealand']:
        params=data[region]['train'][0].region_params[0]
        path=description['regions'][region]['processed_path']
        frame=pd.read_csv(path)
        frame.time=pd.to_datetime(frame.time,utc=True,format='ISO8601')
        frame=frame[frame.time<'2014-01-01'].iloc[256:257+events]
        assert len(frame)==events+1
        times=pd.DatetimeIndex(frame.time).as_unit('ns').asi8.astype(np.float64)/86400e9
        arrival=times[1:]-times[0]
        inter=np.diff(np.r_[0.,arrival,arrival[-1]])
        seq=Sequence(torch.from_numpy(inter),t_nll_start=float(arrival[history-1]),
                     mag=torch.from_numpy(frame.magnitude.to_numpy(np.float64)[1:].copy()))
        sequences.append(attach_region(seq,params,region_id=vocabulary[region]))
    model=build_model('stats_film',0,vocabulary)
    model.train()
    batch=make_batch(sequences)
    started=time.time()
    loss=model.nll_loss(batch,reduction='per_event')['total'].mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.optim.AdamW(model.parameters(),lr=1e-4).step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    torch.cuda.synchronize()
    result=dict(name='project_regional_mamba2',events=events,history=history,batch=3,
                loss=float(loss),gradients_finite=True,optimizer_step_passed=True,elapsed_seconds=time.time()-started)
    print(json.dumps(result),flush=True)
    return result


def main():
    torch.set_num_threads(1)
    from mamba_ssm import Mamba2,Mamba3
    import mamba_ssm
    report=dict(python=sys.executable,python_version=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                compiled_archs=torch.cuda.get_arch_list(),mamba_path=mamba_ssm.__file__,
                mamba_source_commit='e9594ce1c732d97440f0332fdc43170a2294dbfa',checks=[],passed=False)
    report['packages']={name:importlib.metadata.version(name) for name in [
        'mamba-ssm','triton','flash-attn','causal-conv1d','tilelang','quack-kernels','nvidia-cutlass-dsl',
        'nvidia-cuda-nvcc','nvidia-nvvm','apache-tvm-ffi','torch-c-dlpack-ext']}
    report['mamba3_module']=inspect.getfile(Mamba3)
    report['mamba3_module_sha256']=hashlib.sha256(Path(inspect.getfile(Mamba3)).read_bytes()).hexdigest()
    report['cuda_home']=os.environ.get('CUDA_HOME')
    compiler=subprocess.run(['nvcc','--version'],capture_output=True,text=True)
    report['compiler']=compiler.stdout
    output=REPORTS/'environment_6000d_mamba3_20261104.json'
    try:
        x=torch.randn(32,32,device='cuda',requires_grad=True)
        x.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        for architecture,length,mimo in [('mamba2',256,False),('mamba3_siso',256,False),
                                          ('mamba3_siso',2048,False),('mamba3_mimo',256,True)]:
            torch.manual_seed(0)
            x=torch.randn(2,length,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            options=dict(d_model=128,d_state=64,expand=2,headdim=32 if mimo else 64,device='cuda',dtype=torch.bfloat16)
            if architecture=='mamba2':model=Mamba2(**options)
            else:model=Mamba3(**options,is_mimo=mimo,mimo_rank=2 if mimo else 1,chunk_size=16 if mimo else 64)
            report['checks'].append(exercise(architecture,model,x))
            del model,x
            torch.cuda.empty_cache()
            output.write_text(json.dumps(report,indent=2)+'\n')
        for events,history in [(192,64),(3072,1024)]:
            report['checks'].append(project_check(events,history))
            torch.cuda.empty_cache()
        report['passed']=True
    except Exception as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        output.write_text(json.dumps(report,indent=2)+'\n')
    frozen=subprocess.run([sys.executable,'-m','pip','freeze','--local'],capture_output=True,text=True,check=True).stdout
    (REPORTS/'environment_6000d_gpu_packages_20261104.txt').write_text(frozen)
    print('ENVIRONMENT_VALIDATED',str(output),flush=True)


if __name__=='__main__':main()
