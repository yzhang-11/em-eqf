"""Build the three additional catalogs without changing existing cached datasets."""
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data import Catalog
import src.catalogs.research_catalogs_20261106
from src.catalogs.external_catalog import file_sha256
from src.data.preparation_modules.tpp import prepare_data_tpp

STORE=Path('/root/autodl-fs/em_eqf_catalogs/research_20261106')
OUT=ROOT/'experiments/revision/catalog_research_20261106'
NAMES=['USGS_Montana','NORWAY_SW','NORWAY_Bitdalsvatn']


def link(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():assert path.resolve()==target.resolve()
    else:path.symlink_to(target,target_is_directory=target.is_dir())


def main():
    (OUT/'configs').mkdir(parents=True,exist_ok=True)
    download=json.loads((OUT/'reports/download_summary.json').read_text())
    guard=json.loads((ROOT/'experiments/revision/catalog_optimization_20261105/protected_14_before.json').read_text())
    results=[]
    for name in NAMES:
        cls=Catalog.by_name(name+'-Standard');ds=ROOT/'data'/name
        row=download['sources'][name];source=Path(row['artifacts']['quality_screened']['path'])
        assert file_sha256(source)==row['artifacts']['quality_screened']['sha256']
        link(ds/'processed/standard_source_20261106.parquet',source)
        link(ds/'raw/source_snapshot_20261106',STORE/name/'raw')
        cache=STORE/name/'catalogs';cache.mkdir(parents=True,exist_ok=True)
        link(ds/'catalogs/standard_20261106',cache)
        cfg=dict(root_dir=str(ds/'catalogs/standard_20261106'),catalog_file=str(source),
            start_ts=cls.default_start,end_ts=cls.default_end,mag_completeness=cls.default_mc,
            normalize=True,val_start_ts=cls.default_validation,test_start_ts=cls.default_test)
        obj=cls(**cfg);report=json.loads((obj.root_dir/'processing_report.json').read_text())
        scored=report['scored_events'];assert scored['train']>=100 and scored['val']>=20 and scored['test']>=20,(name,scored)
        args=OmegaConf.create(dict(dataset=name,catalog_cfg=cfg,batch_size=2,use_double_precision=True,
            use_all_data=False,minibatch_training=True,max_seq_len=512,mean_nll_events=128,use_b_updater=False))
        _,tr,va,te,_=prepare_data_tpp(args,ds)
        windows={};nonpositive={}
        for split,loader in [('train',tr),('val',va),('test',te)]:
            batch=next(iter(loader));assert torch.isfinite(batch.inter_times).all()
            windows[split]=len(loader.dataset)
            nonpositive[split]=sum(int(((s.inter_times[:-1]<=0)&(s.arrival_times>=s.t_nll_start)).sum()) for s in loader.dataset.sequences)
            assert not nonpositive[split]
        model=OmegaConf.load(ROOT/'config/mixer_tpp_mamba3_rotary.yaml')
        model.dataset=name;model.catalog_cfg=cfg;model.use_double_precision=True;model.use_all_data=False
        model.use_b_updater=False;model.minibatch_training=True;model.max_seq_len=512;model.mean_nll_events=128
        OmegaConf.save(model,OUT/'configs'/f'{name}_mixer_tpp.yaml')
        results.append(dict(dataset=name,cache=str(obj.root_dir),source=str(source),source_sha256=file_sha256(source),
            records=len(obj.full_sequence),mc=cls.default_mc,**scored,start=cls.default_start,end=cls.default_end,
            validation_start=cls.default_validation,test_start=cls.default_test,quality_policy=row['quality_policy'],
            quality_protocol=cls.quality_protocol,windows=windows,nonpositive_scored_intervals=nonpositive,
            artifact_sha256={p.name:file_sha256(p) for p in obj.root_dir.iterdir() if p.is_file()},dataloader_verified=True))
        print('READY',name,len(obj.full_sequence),scored,flush=True)
    assert all(file_sha256(p)==h for p,h in guard.items())
    summary=dict(passed=True,datasets=results,protected_files_unchanged=len(guard),
        note='Research source-specific quality protocols are not equivalent to the operational reviewed/station/gap/RMS protocol.')
    (OUT/'reports/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in results]).to_csv(OUT/'reports/catalogs.csv',index=False)
    print('ALL_RESEARCH_STANDARD_READY',flush=True)


if __name__=='__main__':main()
