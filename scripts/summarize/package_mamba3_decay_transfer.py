"""Archive the user-default Mamba3 experiment and exact reference dependencies."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/root/autodl-tmp/em_eqf');TAG='mamba3_decay_transfer_20261105'
BASE=ROOT/'experiments/revision/multiregion_pretraining';STORE=Path('/root/autodl-fs/em_eqf_training')/TAG
OLD=STORE.parent/'chuandian_scedc_tpp_20261105';REPORTS=BASE/'reports'/TAG
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()

assert read(REPORTS/'final_audit.json')['passed']
assert read(REPORTS/'supplementary_audit.json')['passed']
assert (REPORTS/'master.exit').read_text().strip()=='0'
plan=read(REPORTS/'plan.json');assert sha(OLD/'targets.pt')==plan['data_sha256']
assert sha(STORE/'frozen_runner.py')==plan['runner_sha256']
for x in plan['source_data'].values():assert sha(x['path'])==x['sha256']
rows=[]
for p in sorted((STORE/'runs').glob('*/training_complete.json')):
    d=read(p);assert sha(p.parent/'best_model.pth')==d['checkpoint_sha256']
    row={k:d[k] for k in ['target','family','mode','seed','lr','updates','best_step','parameters','seconds','dataset_sha256','checkpoint_sha256']}
    row.update({f'validation_macro_{k}':v for k,v in d['validation']['macro'].items()})
    row['scored_event_exposures']=sum(d['exposures'].values());row['run_directory']=p.parent.name;rows.append(row)
with (REPORTS/'training_manifest.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
items=[(REPORTS,REPORTS.relative_to(ROOT)),(BASE/'configs'/TAG,Path('experiments/revision/multiregion_pretraining/configs')/TAG),
       (STORE/'runs',Path('experiments/revision/multiregion_pretraining/runs')/TAG),
       (STORE/'frozen_project',Path('frozen_project')),(STORE/'before',Path('before')),
       (BASE/'README.md',Path('experiments/revision/multiregion_pretraining/README.md'))]
for p in STORE.glob('*.py'):items.append((p,Path('frozen_study')/p.name))
for name in ['tuning/run_mamba3_decay_transfer.py','summarize/audit_mamba3_decay_transfer.py','summarize/package_mamba3_decay_transfer.py']:
    items.append((ROOT/'scripts'/name,Path('scripts')/name))
for name in ['targets.pt','sources_before_chuandian_training_end.pt','sources_before_scedc_training_end.pt','frozen_runner.py','frozen_summary.py']:
    items.append((OLD/name,Path('references')/OLD.name/name))
items.append((BASE/'configs'/OLD.name,Path('references')/OLD.name/'configs'))
for tag in [OLD.name,'original_etas_tpp_20261105','recast_single_region_tpp_20261105']:
    items.append((BASE/'reports'/tag,Path('references')/tag/'reports'))
for p in (OLD/'runs').glob('*/training_complete.json'):
    items.append((p,Path('references')/OLD.name/'training_metadata'/p.parent.name/p.name))
# The imported previous control module imports this helper; bootstrap is
# explicitly redirected to the new frozen_project by the current runner.
p=STORE.parent/'mamba3_chuandian_transfer_20261105/frozen_project/scripts/tuning/mamba3_transfer_common.py'
items.append((p,Path('dependencies/mamba3_transfer_common.py')))
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
