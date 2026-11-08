"""Run resumable download -> readiness -> standardization -> independent audit."""
import argparse
import fcntl
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.catalogs.external_catalog import file_sha256
OUT=ROOT/'experiments/revision/catalog_optimization_20261105'
PIPELINE_LOCK=None


def guard():
    protected=json.loads((OUT/'protected_14_before.json').read_text())
    changed=[p for p,h in protected.items() if not Path(p).exists() or file_sha256(p)!=h]
    if changed:raise ValueError(f'Existing protected files changed: {changed}')
    return len(protected)


def status(phase,**extra):
    value=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
    temp=OUT/'pipeline_status.json.part';temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(OUT/'pipeline_status.json');print(json.dumps(value),flush=True)


def run(path,*args):
    subprocess.run([sys.executable,'-u',str(ROOT/path),*args],cwd=ROOT,check=True)


def main():
    global PIPELINE_LOCK
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-existing-only',action='store_true');args=parser.parse_args()
    protected=guard()
    if args.verify_existing_only:
        print('PREFLIGHT_PASSED',protected,flush=True);return
    PIPELINE_LOCK=(OUT/'pipeline.lock').open('a+')
    try:fcntl.flock(PIPELINE_LOCK.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('An optimized NOA pipeline is already running')
    PIPELINE_LOCK.seek(0);PIPELINE_LOCK.truncate();PIPELINE_LOCK.write(str(os.getpid()));PIPELINE_LOCK.flush()
    try:
        status('downloading',protected_files=protected)
        run('scripts/data/resume_noa_catalog.py','--workers','2')
        coverage=json.loads((OUT/'download_complete.json').read_text())
        assert coverage['ready'] and coverage['completed_months']==coverage['expected_months']==144
        source=Path('/root/autodl-fs/em_eqf_catalogs/more_20261105/NOA_Greece/processed/quality_screened_20261105.parquet')
        f=pd.read_parquet(source,columns=['time'])
        t=pd.to_datetime(f.time,utc=True)
        counts=dict(train=int(((t>='2018-01-01')&(t<'2019-01-01')).sum()),
            val=int(((t>='2019-01-01')&(t<'2022-01-01')).sum()),test=int((t>='2022-01-01').sum()))
        readiness=dict(counts=counts,ready=counts['train']>=100 and counts['val']>=20 and counts['test']>=20,
            minimums=dict(train=100,val=20,test=20))
        (OUT/'training_readiness.json').write_text(json.dumps(readiness,indent=2))
        if not readiness['ready']:raise ValueError(f'Insufficient fixed-period strict events: {counts}')
        status('standardizing',counts=counts)
        run('scripts/data/prepare_more_standard.py')
        status('auditing',counts=counts)
        run('experiments/revision/catalog_more_20261105/audit_more_catalogs.py')
        guard()
        reports=ROOT/'experiments/revision/catalog_more_20261105/reports'
        audit=json.loads((reports/'independent_audit.json').read_text())
        summary=json.loads((reports/'summary.json').read_text())
        greek=next(r for r in summary['datasets'] if r['dataset']=='NOA_Greece')
        assert audit['passed'] and audit['total_standard_catalogs']==15
        complete=dict(completed_utc=datetime.now(timezone.utc).isoformat(),passed=True,
            total_standard_catalogs=15,new_dataset='NOA_Greece',records=greek['records'],
            counts=counts,mc=greek['mc'],old_14_files_unchanged=protected,
            coverage_months=144,source_raw_records=coverage['normalised_records'],
            source_strict_records=coverage['strict_records'],cache=greek['cache'],
            cross_catalog_candidate_pairs=audit['cross_catalog_candidate_pairs'])
        (OUT/'pipeline_complete.json').write_text(json.dumps(complete,indent=2),encoding='utf-8')
        status('complete',**complete)
    except Exception:
        status('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
