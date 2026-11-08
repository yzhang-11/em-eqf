"""Reuse the tested standard builder for this batch without editing it."""
import importlib.util
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.catalogs.more_catalogs_20261105

STORE=Path('/root/autodl-fs/em_eqf_catalogs/more_20261105')
OUT=ROOT/'experiments/revision/catalog_more_20261105'
NAMES=['USGS_PNW','USGS_Utah','USGS_NewMadrid','NOA_Greece']


def iceland_readiness(path):
    f=pd.read_parquet(path,columns=['time'])
    t=pd.to_datetime(f.time,utc=True)
    counts=dict(train=int((t<'2020-01-01').sum()),
        val=int(((t>='2020-01-01')&(t<'2020-07-01')).sum()),test=int((t>='2020-07-01').sum()))
    ready=counts['train']>=100 and counts['val']>=20 and counts['test']>=20
    return dict(ready=ready,counts=counts,minimums=dict(train=100,val=20,test=20),
        reason='Usability screen only; do not relax event-type or uncertainty requirements to increase sample size.')


def main():
    path=OUT/'reports/download_summary.json';download=json.loads(path.read_text())
    madrid=json.loads((OUT/'reports/USGS_NewMadrid_audit.json').read_text())
    download['sources']['USGS_NewMadrid']=madrid
    path.write_text(json.dumps(download,indent=2),encoding='utf-8')
    readiness=iceland_readiness(STORE/'ISOR_Hengill/processed/quality_screened_20261105.parquet')
    (OUT/'reports/ISOR_training_readiness.json').write_text(json.dumps(readiness,indent=2))
    spec=importlib.util.spec_from_file_location('_more_standard_builder',ROOT/'scripts/data/prepare_regional_standard.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    names=[n for n in NAMES if n in download['sources']]+(['ISOR_Hengill'] if readiness['ready'] else [])
    builder.STORE=STORE;builder.OUT=OUT;builder.NAMES=names
    builder.main()
    path=OUT/'reports/summary.json';summary=json.loads(path.read_text())
    summary['split']='per-dataset fixed calendar boundaries; Greece starts in 2018 because earlier quality metadata is incomplete; Iceland uses its 2018-2021 observation period'
    summary['notes']=['Previous catalogs and their cache/config files are protected by checksums.',
        'Unknown event types/review states and missing required quality fields fail screening.',
        'The Iceland catalog is explicitly manual ISOR_manual; no SED_auto counterpart is pooled.',
        'Magnitude thresholds are research cutoffs, not certified completeness magnitudes.',
        'Greece pre-2018 records remain in the source audit, outside the model observation window: absent quality metadata must not become a multi-year silent interval.',
        'This batch uses reported horizontal uncertainty, or the reported maximum ellipse axis when scalar uncertainty is absent; confidence definitions are not assumed identical across sources.',
        'Potential cross-catalog duplicates must be reviewed before independent transfer evaluations.']
    summary['ISOR_readiness']=readiness
    summary['staged']=[] if readiness['ready'] else ['ISOR_Hengill: strict subset has insufficient scored events in one or more fixed periods.']
    summary['staged'].extend(f'{n}: {v["reason"]}' for n,v in download.get('staged_downloads',{}).items())
    for row in summary['datasets']:
        cfg=json.loads((Path(row['cache'])/'catalog_cfg.json').read_text())
        row['validation_start']=cfg['val_start_ts'];row['test_start']=cfg['test_start_ts']
    temp=path.with_suffix('.json.part');temp.write_text(json.dumps(summary,indent=2),encoding='utf-8');temp.replace(path)


if __name__=='__main__':main()
