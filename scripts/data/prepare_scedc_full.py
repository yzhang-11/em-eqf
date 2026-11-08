"""Materialize the default SCEDC full-prefix split from audited local annual files."""
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch

BASE=Path('/root/autodl-fs/em_eqf_training/mamba_time_dynamics_20261105/frozen_project')
ORIGIN=Path('/root/autodl-tmp/em_eqf')
STORE=Path('/root/autodl-fs/em_eqf_training/scedc_full_time_dynamics_20261105')
sys.path[:0]=[str(BASE),str(BASE/'scripts/run')]
from src.catalogs.scedc import SCEDC
from src.data import Batch

torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['src/catalogs/scedc.py','src/utils/catalog_utils.py']:
    assert sha(BASE/name)==sha(ORIGIN/name), f'Default catalog implementation changed: {name}'
raw=ORIGIN/'data/SCEDC/raw/scsn_annual_snapshot_20261102'
sources={str(year):raw/f'{year}.catalog' for year in range(1981,2020)}
assert all(p.is_file() for p in sources.values())
STORE.mkdir(parents=True,exist_ok=True)
(STORE/'shared/reports').mkdir(exist_ok=True)
def local_response(url,**kwargs):
    match=re.search(r'/(\d{4})\.catalog$',url)
    assert match and match[1] in sources, url
    return SimpleNamespace(content=sources[match[1]].read_bytes())

with patch('src.catalogs.scedc.requests.get',side_effect=local_response):
    catalog=SCEDC(root_dir=STORE/'catalogs')

assert str(catalog.metadata['train_start_ts'])=='1985-01-01 00:00:00'
assert str(catalog.metadata['val_start_ts'])=='2005-01-01 00:00:00'
assert str(catalog.metadata['test_start_ts'])=='2014-01-01 00:00:00'
assert catalog.metadata['mag_completeness']==2.
sequences={name:getattr(catalog,name)[0] for name in ['train','val','test']}
train=sequences['train']
intervals=train.inter_times[:-1]
stats=dict(tau_mean=float(intervals.mean()),tau_min=float(intervals.min()),tau_max=float(intervals.max()),
    tau_q05=float(intervals.quantile(.5)),tau_q025=float(intervals.quantile(.025)),
    mag_mean=float(train.mag.mean()),time_min=float(train.t_start),time_max=float(train.t_end),
    time_mean=float(train.arrival_times[:-1].mean()),mag_completeness=2.)
stats['richter_b_mle']=math.log10(math.e)/(stats['mag_mean']-2.+.5*catalog.metadata['mag_roundoff_error'])
counts={}
for name,seq in sequences.items():
    assert seq.inter_times.dtype==torch.float64
    assert torch.all(seq.inter_times[:-1]>0)
    assert torch.all(seq.mag>2.)
    batch=Batch.from_list([seq])
    assert int(batch.nll_event_mask.sum())==seq.num_nll_events
    counts[name]=dict(input_events=len(seq),input_tokens=batch.seq_len,scored_events=seq.num_nll_events,
        t_start=float(seq.t_start),t_end=float(seq.t_end),t_nll_start=float(seq.t_nll_start))
for earlier,later in [('train','val'),('val','test')]:
    a,b=sequences[earlier],sequences[later]
    torch.testing.assert_close(a.arrival_times,b.arrival_times[:len(a)],rtol=0,atol=1e-10)
    assert a.t_start==b.t_start==0.
payload=dict(schema='SCEDC-default-full-prefix-f64-v1',dataset='SCEDC',
    sequences={k:s.state_dict() for k,s in sequences.items()},stats=stats,counts=counts,
    metadata={k:str(v) for k,v in catalog.metadata.items()},
    source_sha256={str(p):sha(p) for p in sources.values()},threshold_operator='>',
    default_catalog_signature=str(inspect.signature(SCEDC.__init__)))
path=STORE/'dataset_snapshot.pt'
if path.exists():raise RuntimeError('Existing SCEDC snapshot; do not replace it')
torch.save(payload,path)
report={k:v for k,v in payload.items() if k!='sequences'}
report['dataset_sha256']=sha(path)
report['catalog_cache']=str(catalog.root_dir)
(STORE/'reports/data_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(dataset=str(path),sha256=sha(path),counts=counts,stats=stats),indent=2),flush=True)
