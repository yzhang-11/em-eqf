"""Frozen model code and paired batches for the regional capacity study."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT=Path('/root/autodl-tmp/em_eqf')
STUDY=ROOT/'experiments/revision/multiregion_pretraining'
REPORTS=STUDY/'reports/mamba3_oracle_residual_20261105'
CONFIGS=STUDY/'configs/mamba3_oracle_residual_20261105'
STORE=Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105')
RUNS=STUDY/'runs/mamba3_oracle_residual_20261105'
FROZEN=STORE/'frozen_project'
DATA=ROOT/'data/MultiRegion/catalogs/region_capacity_20261105.pt'
SIZES={'small':(64,2),'large':(128,4)}
LOCAL_STEPS=800
SEEDS=[0,1,2]
_TRAIN_CACHE={}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def freeze():
    """Called once, before workers start; no shared model source is edited."""
    assert not (REPORTS/'frozen_source.json').exists()
    paths=list((ROOT/'src').rglob('*.py'))+[ROOT/p for p in [
        'scripts/run/run_multidomain_pretraining.py','scripts/run/run_multiregion_pretraining.py',
        'scripts/tuning/region_capacity_common.py','scripts/tuning/run_region_capacity_study.py',
        'scripts/tuning/mamba3_transfer_common.py','scripts/tuning/run_mamba3_transfer_pretraining.py',
        'scripts/data/prepare_region_capacity_data.py',
        'experiments/revision/main/configs/pretrain_azdx_main_seed_0.yaml']]
    manifest={}
    for source in paths:
        relative=source.relative_to(ROOT);target=FROZEN/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        manifest[str(relative)]=sha(target)
    # Preserve the reference's Mamba2 forward computation. The live checkout
    # also contains later inference-alignment changes from another experiment.
    relative='src/models/mamba/mamba2_rotary.py'
    previous=Path('/root/autodl-fs/em_eqf_training/region_capacity_20261105/frozen_project')/relative
    shutil.copy2(previous,FROZEN/relative);manifest[relative]=sha(FROZEN/relative)
    save(REPORTS/'frozen_source.json',manifest)
    (STORE/'runs').mkdir(parents=True,exist_ok=True)
    RUNS.parent.mkdir(parents=True,exist_ok=True)
    if not RUNS.exists():RUNS.symlink_to(STORE/'runs',target_is_directory=True)


def bootstrap():
    manifest=json.loads((REPORTS/'frozen_source.json').read_text())
    assert all(sha(FROZEN/f)==h for f,h in manifest.items())
    sys.path[:0]=[str(FROZEN),str(FROZEN/'scripts/run')]
    import src
    assert Path(src.__file__).resolve().is_relative_to(FROZEN.resolve())


def backbone(size):
    from run_multidomain_pretraining import model_settings
    cfg=model_settings('stats_film');width,depth=SIZES[size]
    cfg.d_model=width
    cfg.mixer_model_config.d_model=width
    cfg.mixer_model_config.n_layer=depth
    cfg.mixer_model_config.d_intermediate=2*width
    cfg.mixer_model_config.input_proj_hidden=2*width
    cfg.resume_path=None;cfg.load_specific_parts=[]
    return OmegaConf.to_container(cfg,resolve=True)


def sequence(spec,region_id,first,targets,context):
    from src.data import Sequence
    from src.models.tpp.region_mixer_tpp import attach_region
    context=min(context,first-1)
    assert context>=1 and first+targets<=len(spec['ns'])
    left=first-context;right=first+targets
    arrival=(spec['ns'][left:right]-spec['ns'][left-1])/86400e9
    inter=np.diff(np.r_[0.,arrival,arrival[-1]])
    seq=Sequence(torch.from_numpy(inter),t_nll_start=float(arrival[context-1]),
                 mag=torch.from_numpy(spec['magnitude'][left:right].copy()),
                 event_index=torch.arange(left,right,dtype=torch.long))
    # Match the exact torch cumulative sum used by Batch. A NumPy boundary one
    # ULP below that sum can otherwise score the last context event as a target.
    seq.t_nll_start=float(seq.arrival_times[context-1])
    seq=attach_region(seq,spec['metadata'])
    seq.region_id=torch.full((len(seq),),region_id,dtype=torch.long)
    return seq


def training_batch(data,region,block,length):
    from run_multidomain_pretraining import make_batch
    key=(region,block,length)
    if key not in _TRAIN_CACHE:
        spec=data['regions'][region];region_id=data['vocabulary'][region]
        first=spec['block_starts'][block];targets=spec['block_targets'][block]
        items=([sequence(spec,region_id,first,targets,1024)] if length=='long' else
               [sequence(spec,region_id,first+offset,min(128,targets-offset),64) for offset in range(0,targets,128)])
        _TRAIN_CACHE[key]=items
    items=_TRAIN_CACHE[key]
    batch=make_batch(items)
    assert int(batch.nll_event_mask.sum())==data['regions'][region]['block_targets'][block]
    return batch


def evaluation_pools(data,split,context=1024):
    pools={}
    for region,spec in data['regions'].items():
        low,high=(spec['train_end'],spec['val_end']) if split=='val' else (spec['val_end'],spec['test_end'])
        pools[region]=[sequence(spec,data['vocabulary'][region],first,min(128,high-first),context)
                       for first in range(low,high,128)]
    return pools


@torch.no_grad()
def evaluate(model,pools):
    """Event-weighted per region; equal-region macro. Partial tails retained."""
    from run_multidomain_pretraining import make_batch
    model.eval();regions={};windows={}
    for region,items in pools.items():
        totals=dict.fromkeys(['time','mag','total'],0.)
        count=0;rows=[]
        for left in range(0,len(items),8):
            batch=make_batch(items[left:left+8]);loss=model.nll_loss(batch,reduction='per_event')
            n=batch.nll_event_mask.sum(-1)
            for k in totals:totals[k]+=float((loss[k]*n).sum())
            count+=int(n.sum())
            for j in range(len(n)):
                rows.append(dict(events=int(n[j]),**{k:float(loss[k][j]) for k in totals}))
        assert count>0
        regions[region]={k:v/count for k,v in totals.items()}
        assert all(np.isfinite(v) for v in regions[region].values())
        windows[region]=rows
    macro={k:float(np.mean([v[k] for v in regions.values()])) for k in totals}
    return dict(regions=regions,macro=macro),windows


def state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
