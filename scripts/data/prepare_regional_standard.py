"""Build compatible caches/configs for the four audited regional additions."""
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data import Catalog
from src.catalogs.external_catalog import file_sha256
import src.catalogs.regional_catalogs_20261105
from src.data.preparation_modules.tpp import prepare_data_tpp

STORE=Path('/root/autodl-fs/em_eqf_catalogs/additions_20261105')
OUT=ROOT/'experiments/revision/catalog_additions_20261105'
NAMES=['USGS_Hawaii','USGS_PuertoRico','USGS_Alaska','SED_Switzerland']


def link(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.resolve()!=target.resolve():raise FileExistsError(path)
    else:path.symlink_to(target,target_is_directory=target.is_dir())


def main():
    (OUT/'configs').mkdir(parents=True,exist_ok=True)
    download=json.loads((OUT/'reports/download_summary.json').read_text())
    if download['errors']:raise ValueError(download['errors'])
    before=json.loads((OUT/'reports/protected_before.json').read_text())
    results=[]
    for name in NAMES:
        ds=ROOT/'data'/name;source=STORE/name/'processed/quality_screened_20261105.parquet'
        expected=download['sources'][name]['artifacts']['quality_screened']['sha256']
        assert file_sha256(source)==expected
        link(ds/'raw/source_snapshot_20261105',STORE/name/'raw')
        link(ds/'processed/standard_source_20261105.parquet',source)
        cache=STORE/name/'catalogs';cache.mkdir(parents=True,exist_ok=True)
        link(ds/'catalogs/standard_20261105',cache)
        cls=Catalog.by_name(name+'-Standard')
        cfg=dict(root_dir=str(ds/'catalogs/standard_20261105'),catalog_file=str(ds/'processed/standard_source_20261105.parquet'),
            start_ts=cls.default_start,end_ts=cls.default_end,mag_completeness=cls.default_mc,normalize=True,
            val_start_ts=cls.default_validation,test_start_ts=cls.default_test)
        obj=cls(**cfg)
        report=json.loads((obj.root_dir/'processing_report.json').read_text())
        target_sets={}
        for split,part in [('train',obj.train),('val',obj.val),('test',obj.test)]:
            seq=part[0];mask=seq.arrival_times>=seq.t_nll_start
            ids=seq.event_index[mask].to(torch.int64).tolist()
            assert len(ids)==len(set(ids))==report['scored_events'][split]
            target_sets[split]=set(ids)
        assert not target_sets['train']&target_sets['val']
        assert not target_sets['train']&target_sets['test']
        assert not target_sets['val']&target_sets['test']
        args=OmegaConf.create(dict(dataset=name,catalog_cfg=cfg,batch_size=2,use_double_precision=True,
            use_all_data=False,minibatch_training=True,max_seq_len=512,mean_nll_events=128,use_b_updater=False))
        _,tr,va,te,_=prepare_data_tpp(args,ds)
        windows,zero_scored={},{}
        for split,loader in [('train',tr),('val',va),('test',te)]:
            batch=next(iter(loader));assert torch.isfinite(batch.inter_times).all()
            windows[split]=len(loader.dataset)
            zero_scored[split]=sum(int(((s.inter_times[:-1]<=0)&(s.arrival_times>=s.t_nll_start)).sum())
                                   for s in loader.dataset.sequences)
            assert zero_scored[split]==0,(name,split,'Nonpositive scored interval')
        model=OmegaConf.load(ROOT/'config/mixer_tpp_mamba3_rotary.yaml')
        model.dataset=name;model.catalog_cfg=cfg;model.use_double_precision=True;model.use_all_data=False
        model.use_b_updater=False;model.minibatch_training=True;model.max_seq_len=512;model.mean_nll_events=128
        OmegaConf.save(model,OUT/'configs'/f'{name}_mixer_tpp.yaml')
        artifacts={p.name:file_sha256(p) for p in obj.root_dir.iterdir() if p.is_file()}
        row=dict(dataset=name,provider=name,cache=str(obj.root_dir),source=str(source),source_sha256=expected,
            records=len(obj.full_sequence),mc=cls.default_mc,start=cls.default_start,end=cls.default_end,
            **report['scored_events'],windows=windows,nonpositive_scored_intervals=zero_scored,
            quality_policy=download['sources'][name]['quality_policy'],artifact_sha256=artifacts,dataloader_verified=True)
        results.append(row)
        pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in results]).to_csv(OUT/'reports/catalogs.csv',index=False)
        print('STANDARD_READY',name,row['records'],report['scored_events'],flush=True)
    after={p:file_sha256(p) for p in before}
    assert before==after,'A protected existing artifact changed'
    summary=dict(datasets=results,passed=True,protected_unchanged=True,before=before,after=after,
        source_references={k:v['source']['reference'] for k,v in download['sources'].items()},
        split='training 2014-2018, validation 2019-2021, testing 2022-2025; separate from earlier experiments',
        notes=['All normalization and reference statistics use training scored events only.',
               'Validation/test sequences retain history; scored event sets are disjoint.',
               'Fixed magnitude cuts do not certify catalog completeness.',
               'Alaska has missing horizontal-quality fields; passing its documented RMS/depth screen does not establish high horizontal accuracy.',
               'SED regional events may overlap the existing INGV catalog; cross-catalog independence requires a separate experiment protocol.'])
    (OUT/'reports/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('ALL_STANDARD_READY',flush=True)


if __name__=='__main__':main()

