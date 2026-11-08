"""Publish standard Catalog artifacts from the audited public snapshots."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data import Catalog, TppDataset
from src.catalogs.external_catalog import file_sha256
from src.data.preparation_modules.tpp import prepare_data_tpp

SNAPSHOT=ROOT/'data/CatalogExpansion_20261105'
STORE=Path('/root/autodl-fs/em_eqf_catalogs/standard_20261105')
OUT=ROOT/'experiments/revision/catalog_standardization_20261105'
DATASETS={
    'JMA_Kyushu':('JMA','japan_kyushu_quality_screened_20261105.parquet'),
    'JMA_WestHonshu':('JMA','japan_west_honshu_quality_screened_20261105.parquet'),
    'JMA_EastHonshu':('JMA','japan_east_honshu_quality_screened_20261105.parquet'),
    'JMA_Hokkaido':('JMA','japan_hokkaido_quality_screened_20261105.parquet'),
    'GeoNet':('GeoNet','quality_screened_20261105.parquet'),
    'INGV':('INGV','quality_screened_20261105.parquet'),
    'NCEDC':('NCEDC','quality_screened_20261105.parquet'),
}


def link(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.resolve()!=target.resolve():
            raise FileExistsError(f'Refusing to replace {path}')
    else:
        path.symlink_to(target,target_is_directory=target.is_dir())


def main():
    (OUT/'shared/reports').mkdir(parents=True,exist_ok=True)
    (OUT/'configs').mkdir(parents=True,exist_ok=True)
    protected=[ROOT/'data/MultiRegion/catalogs/pilot_2007_2019.pt',
               ROOT/'data/MultiRegion/catalogs/long_sequence_20261105.pt',
               ROOT/'src/catalogs/scedc.py',ROOT/'src/catalogs/chuandian.py',ROOT/'config/data_tpp.yaml']
    before={str(p):file_sha256(p) for p in protected if p.exists()}
    results,configs=[],{}
    for name,(provider,source_name) in DATASETS.items():
        ds=ROOT/'data'/name
        source=SNAPSHOT/provider/'processed'/source_name
        if not source.exists():
            raise FileNotFoundError(source)
        upstream=ROOT/'data/JMA/raw' if provider=='JMA' else SNAPSHOT/provider/'raw'
        link(ds/'raw/source_snapshot_20261105',upstream)
        link(ds/'processed/standard_source_20261105.parquet',source)
        cache_store=STORE/name/'catalogs'
        cache_store.mkdir(parents=True,exist_ok=True)
        link(ds/'catalogs/standard_20261105',cache_store)
        cls=Catalog.by_name(f'{name}-Standard')
        obj=cls(data_dir=ds)
        frame=pd.read_parquet(obj.root_dir/'catalog.parquet')
        report=json.loads((obj.root_dir/'processing_report.json').read_text())
        assert report['records']==len(obj.full_sequence)==len(frame)
        assert obj.full_sequence.inter_times.dtype==torch.float64
        assert torch.all(obj.full_sequence.inter_times[:-1]>0)
        assert frame.event_id.is_unique
        target_ids={}
        for split,ds_part in [('train',obj.train),('val',obj.val),('test',obj.test)]:
            seq=ds_part[0]
            ids=seq.event_index[seq.arrival_times>=seq.t_nll_start].to(torch.long).tolist()
            assert len(ids)==report['scored_events'][split],(name,split,len(ids),report['scored_events'])
            target_ids[split]=set(ids)
        for a,b in [('train','val'),('train','test'),('val','test')]:
            assert not target_ids[a]&target_ids[b]
        # Exercise the actual project data-preparation entrypoint and collator.
        args=OmegaConf.create(dict(dataset=name,catalog_cfg={
            'root_dir':str(ds/'catalogs/standard_20261105'),
            'catalog_file':str(ds/'processed/standard_source_20261105.parquet')},batch_size=2,
            use_double_precision=True,use_all_data=False,minibatch_training=True,
            max_seq_len=512,mean_nll_events=128,use_b_updater=False))
        _,train_loader,val_loader,test_loader,reloaded=prepare_data_tpp(args,ds)
        # split_minibatches returns a DummyCatalog wrapper rather than the
        # constructor object; provenance remains available in its metadata.
        assert reloaded.metadata['source_sha256']==obj.metadata['source_sha256']
        windows={}
        for split,loader in [('train',train_loader),('val',val_loader),('test',test_loader)]:
            batch=next(iter(loader))
            assert torch.isfinite(batch.inter_times).all()
            windows[split]=len(loader.dataset)
        cfg=dict(root_dir=str(ds/'catalogs/standard_20261105'),catalog_file=str(ds/'processed/standard_source_20261105.parquet'),
                 mag_completeness=cls.default_mc,normalize=True,start_ts=cls.default_start,end_ts=cls.default_end,
                 val_start_ts='2014-01-01',test_start_ts='2016-01-01')
        configs[name]=cfg
        base=OmegaConf.load(ROOT/'config/mixer_tpp_mamba3_rotary.yaml')
        base.dataset=name
        base.catalog_cfg=cfg
        base.use_double_precision=True
        base.use_all_data=False
        base.use_b_updater=False
        base.minibatch_training=True
        base.max_seq_len=512
        base.mean_nll_events=128
        # Test configuration only; training is not launched by this script.
        OmegaConf.save(base,OUT/'configs'/f'{name}_mixer_tpp.yaml')
        result=dict(dataset=name,provider=provider,cache=str(obj.root_dir),source=str(source.resolve()),
                    source_sha256=file_sha256(source),records=len(frame),**report['scored_events'],
                    windows=windows,tied_rows=report['tied_rows_retained'],mc=cls.default_mc,
                    normalization='training-only',dataloader_verified=True,
                    artifact_sha256={p.name:file_sha256(p) for p in obj.root_dir.iterdir() if p.is_file()})
        results.append(result)
        print('PREPARED',name,'events',len(frame),'splits',report['scored_events'],'windows',windows,flush=True)
        pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,dict)} for x in results]).to_csv(OUT/'reports/catalogs.csv',index=False)
    after={p:file_sha256(p) for p in before}
    assert before==after,'Existing data or catalog implementation changed during preparation'
    summary=dict(datasets=results,passed=True,protected_unchanged=True,before=before,after=after,
                 staged=['IPOCDeep: absolute-time convention is not confirmed'],
                 time_unit='days',split='training <2014; validation 2014-2015; testing >=2016 through source end',
                 caveats=['Fixed Mc cutoffs are research choices, not completeness certifications.',
                          'GeoNet CSV/FDSN discrepancy remains documented in the upstream snapshot audit.',
                          'NCEDC may overlap Geysers and bordering SCEDC records; cross-dataset independence is not established.',
                          'Validation and test sequences contain past context; their scored NLL event sets are disjoint.'])
    (OUT/'reports/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    OmegaConf.save(OmegaConf.create(dict(dataset_catalog_cfg=configs,catalog_cfg='${dataset_catalog_cfg.${dataset}}')),
                   ROOT/'config/data_external_tpp_20261105.yaml')
    print('ALL_PREPARED',OUT,flush=True)


if __name__=='__main__':
    main()

