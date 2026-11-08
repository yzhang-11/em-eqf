"""Preserve the original-implementation ETAS fits and their exact dependencies."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/root/autodl-tmp/em_eqf');TAG='original_etas_tpp_20261105'
BASE=ROOT/'experiments/revision/multiregion_pretraining';STORE=Path('/root/autodl-fs/em_eqf_training')/TAG
OLD=STORE.parent/'chuandian_scedc_tpp_20261105';REPORTS=BASE/'shared/reports'/TAG
REFERENCE=STORE.parent/'mamba3_chuandian_transfer_20261105/frozen_project'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()

assert read(REPORTS/'final_audit.json')['passed']
assert (REPORTS/'master.exit').read_text().strip()=='0'
plan=read(REPORTS/'plan.json');assert sha(OLD/'targets.pt')==plan['data_sha256']
assert sha(STORE/'frozen_runner.py')==plan['runner_sha256']
assert all(sha(ROOT/name)==value for name,value in plan['native_source'].items())
rows=[]
for p in (STORE/'runs').glob('*/training_complete.json'):
    d=read(p);assert sha(p.parent/'best_model.pth')==d['checkpoint_sha256']
    rows.append({k:d[k] for k in ['target','epochs','best_epoch','train_events','seconds','checkpoint_sha256']})
with (REPORTS/'training_manifest.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
items=[(REPORTS,REPORTS.relative_to(ROOT)),(BASE/'configs'/TAG,Path('experiments/revision/multiregion_pretraining/configs')/TAG),
       (STORE/'runs',Path('experiments/revision/multiregion_pretraining/runs')/TAG),
       (BASE/'README.md',Path('experiments/revision/multiregion_pretraining/README.md')),
       (REFERENCE,Path('frozen_project'))]
for p in STORE.glob('*.py'):items.append((p,Path('frozen_study')/p.name))
for name in ['tuning/run_original_etas_tpp.py','summarize/package_original_etas_tpp.py']:
    items.append((ROOT/'scripts'/name,Path('scripts')/name))
for name in ['targets.pt','frozen_runner.py']:items.append((OLD/name,Path('references')/OLD.name/name))
for name in ['summary.json','data.json','etas/test_results.json']:
    items.append((BASE/'shared/reports'/OLD.name/name,Path('references')/OLD.name/'shared/reports'/name))
for source,name in [(BASE/'reports/mamba3_oracle_residual_20261105/frozen_source.json','frozen_model_manifest.json'),
                    (BASE/'configs/mamba3_oracle_residual_20261105/study_plan.json','original_backbone_plan.json')]:
    items.append((source,Path('dependencies')/name))
manifest={};restore={};generated={REPORTS/'bundle_manifest.json',REPORTS/'restore_paths.json'}
for source,relative in items:
    for p in source.rglob('*') if source.is_dir() else [source]:
        if not p.is_file() or p.is_symlink() or p in generated:continue
        name=relative/p.relative_to(source) if source.is_dir() else relative
        manifest[str(name)]=dict(bytes=p.stat().st_size,sha256=sha(p));restore[str(name)]=str(p)
(REPORTS/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
(REPORTS/'restore_paths.json').write_text(json.dumps(restore,indent=2))
archive=STORE/(TAG+'.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for source,relative in items:tar.add(source,arcname=str(relative))
meta=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),regular_files=len(manifest),fits=len(rows))
(STORE/'bundle.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
