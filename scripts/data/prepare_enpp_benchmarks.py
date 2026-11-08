"""Materialize seven official EarthquakeNPP variants plus legacy Japan separately."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data import Catalog
from src.catalogs.earthquakenpp_20261106 import SPECS,COMMIT,read_author_csv
from src.catalogs.external_catalog import file_sha256
from src.data.preparation_modules.tpp import prepare_data_tpp

STORE=Path('/root/autodl-fs/em_eqf_catalogs/earthquakenpp_20261106')
OUT=ROOT/'experiments/revision/catalog_benchmarks_20261106'


def link(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():assert path.resolve()==target.resolve()
    else:path.symlink_to(target,target_is_directory=target.is_dir())


def main():
    (OUT/'configs').mkdir(parents=True,exist_ok=True);(OUT/'reports').mkdir(parents=True,exist_ok=True)
    guard=json.loads((ROOT/'experiments/revision/catalog_optimization_20261105/protected_14_before.json').read_text())
    results=[]
    for name,s in SPECS.items():
        cls=Catalog.by_name(name+'-Standard');ds=ROOT/'data'/name
        source=STORE/'author_snapshot/Datasets'/s['file']
        link(ds/'raw/author_snapshot',source.parent)
        cache=STORE/'catalogs'/name;cache.mkdir(parents=True,exist_ok=True)
        link(ds/'catalogs/author_20261106',cache)
        cfg=dict(root_dir=str(ds/'catalogs/author_20261106'),catalog_file=str(source),mag_completeness=s['mc'],
            normalize=True,start_ts=s['start'],end_ts=s['end'],train_start_ts=s['train'],val_start_ts=s['val'],test_start_ts=s['test'])
        obj=cls(**cfg);report=json.loads((obj.root_dir/'processing_report.json').read_text())
        frame=pd.read_parquet(obj.root_dir/'catalog.parquet');raw=read_author_csv(source)
        raw=raw.set_index('id').loc[frame.id]
        for col in ['time','magnitude','latitude','longitude','x','y']:
            assert np.array_equal(frame[col].to_numpy(),raw[col].to_numpy()),(name,col)
        targets={};expected=report['scored_events']
        for split,part in [('train',obj.train),('val',obj.val),('test',obj.test)]:
            seq=part[0];ids=seq.event_index[seq.arrival_times>=seq.t_nll_start].to(torch.int64).tolist()
            assert len(ids)==len(set(ids))==expected[split],(name,split,len(ids),expected[split])
            targets[split]=set(ids)
        assert not targets['train']&targets['val'] and not targets['train']&targets['test'] and not targets['val']&targets['test']
        assert 'depth' not in obj.full_sequence and obj.full_sequence.inter_times.dtype==torch.float64
        args=OmegaConf.create(dict(dataset=name,catalog_cfg=cfg,batch_size=2,use_double_precision=True,
            use_all_data=False,minibatch_training=True,max_seq_len=512,mean_nll_events=128,use_b_updater=False))
        _,tr,va,te,_=prepare_data_tpp(args,ds)
        windows={};bad={}
        for split,loader in [('train',tr),('val',va),('test',te)]:
            batch=next(iter(loader));assert torch.isfinite(batch.inter_times).all()
            windows[split]=len(loader.dataset)
            bad[split]=sum(int(((seq.inter_times[:-1]<=0)&(seq.arrival_times>=seq.t_nll_start)).sum()) for seq in loader.dataset.sequences)
            assert not bad[split],(name,split,'Nonpositive scored interval')
        model=OmegaConf.load(ROOT/'config/mixer_tpp_mamba3_rotary.yaml')
        model.dataset=name;model.catalog_cfg=cfg;model.use_double_precision=True;model.use_all_data=False
        model.use_b_updater=False;model.minibatch_training=True;model.max_seq_len=512;model.mean_nll_events=128
        OmegaConf.save(model,OUT/'configs'/f'{name}_mixer_tpp.yaml')
        row=dict(dataset=name,official_benchmark=not s.get('deprecated',False),compatibility_only=s.get('deprecated',False),
            cache=str(obj.root_dir),source=str(source),source_sha256=file_sha256(source),records=len(frame),mc=s['mc'],
            **expected,start=s['start'],train_start=s['train'],validation_start=s['val'],test_start=s['test'],end=s['end'],
            author_commit=COMMIT,depth_available=False,loc_coordinates='author projected x/y km; training-only minmax',
            windows=windows,nonpositive_scored_intervals=bad,dataloader_verified=True,
            artifact_sha256={p.name:file_sha256(p) for p in obj.root_dir.iterdir() if p.is_file()})
        results.append(row);print('BENCHMARK_READY',name,row['records'],expected,flush=True)
    assert all(file_sha256(p)==h for p,h in guard.items())
    result=dict(passed=True,author_commit=COMMIT,datasets=results,official_benchmarks=7,deprecated_compatibility=1,
        original_14_protected_files_unchanged=len(guard),
        caveats=['Same-region and different-Mc variants are not independent earthquake samples.',
            'Japan_Deprecated is not an official EarthquakeNPP benchmark and is retained only for old-paper compatibility.',
            'Focal depth is absent from author CSV files; depth marks are omitted.',
            'Only x/y normalization uses scored training events. Auxiliary history is retained as historical context.'])
    (OUT/'reports/summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in results]).to_csv(OUT/'reports/benchmark_catalogs.csv',index=False)
    print('ALL_BENCHMARKS_READY',flush=True)


if __name__=='__main__':main()
