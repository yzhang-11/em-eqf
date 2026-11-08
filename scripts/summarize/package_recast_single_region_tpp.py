"""Archive RECAST fits and the exact frozen reference inputs they consume."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/root/autodl-tmp/em_eqf');TAG='recast_single_region_tpp_20261105'
BASE=ROOT/'experiments/revision/multiregion_pretraining'
STORE=Path('/root/autodl-fs/em_eqf_training')/TAG;OLD=STORE.parent/'chuandian_scedc_tpp_20261105'
REPORTS=BASE/'reports'/TAG
REFERENCE=STORE.parent/'mamba3_chuandian_transfer_20261105/frozen_project'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


assert read(REPORTS/'final_audit.json')['passed']
assert read(REPORTS/'numerical_floor_audit.json')['passed']
assert (REPORTS/'master.exit').read_text().strip()=='0'
plan=read(REPORTS/'plan.json')
assert sha(OLD/'targets.pt')==plan['data_sha256']
assert all(sha(path)==value for path,value in plan['references'].items())
rows=[]
for p in sorted((STORE/'runs').glob('*/training_complete.json')):
    d=read(p);assert sha(p.parent/'best_model.pth')==d['checkpoint_sha256']
    row={k:d[k] for k in ['target','family','seed','lr','updates','best_step','parameters','seconds','dataset_sha256','checkpoint_sha256']}
    row.update({f'validation_{k}':v for k,v in d['validation']['macro'].items()})
    row['scored_event_exposures']=sum(d['exposures'].values());row['run_directory']=p.parent.name;rows.append(row)
with (REPORTS/'training_manifest.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
items=[(REPORTS,REPORTS.relative_to(ROOT)),(BASE/'configs'/TAG,Path('experiments/revision/multiregion_pretraining/configs')/TAG),
       (STORE/'runs',Path('experiments/revision/multiregion_pretraining/runs')/TAG),
       (BASE/'README.md',Path('experiments/revision/multiregion_pretraining/README.md')),
       (REFERENCE,Path('frozen_project'))]
for p in STORE.glob('*.py'):items.append((p,Path('frozen_study')/p.name))
for name in ['tuning/run_recast_single_region_tpp.py','summarize/audit_recast_single_region_tpp.py','summarize/package_recast_single_region_tpp.py']:
    items.append((ROOT/'scripts'/name,Path('scripts')/name))
ref=Path('references')/OLD.name
for name in ['targets.pt','frozen_runner.py','frozen_summary.py']:
    items.append((OLD/name,ref/name))
items.extend([(BASE/'reports'/OLD.name,ref/'reports'),(BASE/'configs'/OLD.name,ref/'configs')])
for p in (OLD/'runs').glob('*/training_complete.json'):
    items.append((p,ref/'training_metadata'/p.parent.name/p.name))
for source,name in [(BASE/'reports/mamba3_oracle_residual_20261105/frozen_source.json','frozen_model_manifest.json'),
                    (BASE/'configs/mamba3_oracle_residual_20261105/study_plan.json','original_backbone_plan.json')]:
    items.append((source,Path('dependencies')/name))
manifest={};restore={};generated={REPORTS/'bundle_manifest.json',REPORTS/'restore_paths.json'}
for source,relative in items:
    files=source.rglob('*') if source.is_dir() else [source]
    for p in files:
        if not p.is_file() or p.is_symlink() or p in generated:continue
        name=relative/p.relative_to(source) if source.is_dir() else relative
        manifest[str(name)]=dict(bytes=p.stat().st_size,sha256=sha(p));restore[str(name)]=str(p)
(REPORTS/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
(REPORTS/'restore_paths.json').write_text(json.dumps(restore,indent=2))
archive=STORE/(TAG+'.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for source,relative in items:tar.add(source,arcname=str(relative))
meta=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),regular_files=len(manifest),recast_fits=len(rows))
(STORE/'bundle.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
