"""Preserve target TPP data, checkpoints, frozen source, and exact hashes."""
from pathlib import Path
import csv
import hashlib
import json
import tarfile

ROOT=Path('/root/autodl-tmp/em_eqf')
TAG='chuandian_scedc_tpp_20261105'
STORE=Path('/root/autodl-fs/em_eqf_training')/TAG
BASE=ROOT/'experiments/revision/multiregion_pretraining'
REPORTS=BASE/'reports'/TAG
REFERENCE=Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105/frozen_project')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


assert json.loads((REPORTS/'final_audit.json').read_text())['passed']
assert json.loads((REPORTS/'data_integrity_audit.json').read_text())['passed']
assert json.loads((REPORTS/'numerical_floor_audit.json').read_text())['passed']
assert (REPORTS/'finalization.exit').read_text().strip()=='0'
assert (REPORTS/'numerics.exit').read_text().strip()=='0'
training=[]
for p in sorted((STORE/'runs').glob('*/training_complete.json')):
    result=json.loads(p.read_text())
    row={k:result[k] for k in ['target','family','seed','mode','lr','updates','best_step','parameters','seconds','dataset_sha256','checkpoint_sha256']}
    row.update({f'validation_macro_{k}':v for k,v in result['validation']['macro'].items()})
    row['scored_event_exposures']=sum(result['exposures'].values())
    row['run_directory']=p.parent.name
    training.append(row)
with (REPORTS/'training_manifest.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(training[0]));writer.writeheader();writer.writerows(training)
items=[(REPORTS,REPORTS.relative_to(ROOT)),
       (BASE/'README.md',BASE.relative_to(ROOT)/'README.md'),
       (BASE/'configs'/TAG,Path('experiments/revision/multiregion_pretraining/configs')/TAG),
       (STORE/'runs',Path('experiments/revision/multiregion_pretraining/runs')/TAG),
       (REFERENCE,Path('frozen_project'))]
for p in STORE.iterdir():
    if p.is_file() and p.suffix in ['.pt','.py']:
        items.append((p,Path('frozen_study')/p.name))
for name in ['data/prepare_chuandian_scedc_tpp.py','tuning/run_chuandian_scedc_tpp.py',
             'summarize/summarize_chuandian_scedc_tpp.py','summarize/package_chuandian_scedc_tpp.py',
             'summarize/audit_chuandian_scedc_tpp_numerics.py']:
    p=ROOT/'scripts'/name;items.append((p,Path('scripts')/name))
dependency=BASE/'reports/mamba3_oracle_residual_20261105/frozen_source.json'
items.append((dependency,Path('dependencies/frozen_model_manifest.json')))
dependency=BASE/'configs/mamba3_oracle_residual_20261105/study_plan.json'
items.append((dependency,Path('dependencies/original_backbone_plan.json')))
manifest={};restore={}
for source,relative in items:
    files=source.rglob('*') if source.is_dir() else [source]
    for p in files:
        if not p.is_file() or p.is_symlink() or p.name in ['bundle_manifest.json','restore_paths.json']:continue
        name=relative/p.relative_to(source) if source.is_dir() else relative
        manifest[str(name)]=dict(bytes=p.stat().st_size,sha256=sha(p))
        restore[str(name)]=str(p)
(REPORTS/'restore_paths.json').write_text(json.dumps(restore,indent=2))
(REPORTS/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
archive=STORE/(TAG+'.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for source,relative in items:tar.add(source,arcname=str(relative))
meta=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),regular_files=len(manifest))
(STORE/'bundle.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta))
