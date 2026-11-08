"""Export audited settings, reports, selected checkpoints and frozen source."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path('/root/autodl-tmp/em_eqf')
REL=Path('experiments/revision/multiregion_pretraining')
REPORTS=ROOT/REL/'reports/region_capacity_20261105'
STORE=Path('/root/autodl-fs/em_eqf_training/region_capacity_20261105')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert json.loads((REPORTS/'final_audit.json').read_text())['passed']
    assert (REPORTS/'results_zh.md').exists()
    entries={}
    def add(source,relative):
        source=Path(source)
        if source.is_file():entries[str(relative)]=source
    for branch in ['configs/region_capacity_20261105','reports/region_capacity_20261105']:
        for source in (ROOT/REL/branch).rglob('*'):
            if source.suffix in {'.json','.md','.csv','.xml','.log'}:
                add(source,source.relative_to(ROOT))
    for source in (STORE/'runs').rglob('*'):
        if source.name=='best_model.pth' or source.suffix=='.json':
            add(source,REL/'runs/region_capacity_20261105'/source.relative_to(STORE/'runs'))
    for branch in ['frozen_project','frozen_etas','convergence_controls']:
        for source in (STORE/branch).rglob('*'):
            if source.suffix in {'.py','.yaml','.json'} or source.name=='best_model.pth':
                add(source,Path(branch)/source.relative_to(STORE/branch))
    add(STORE/'region_capacity_20261105.pt','data/MultiRegion/catalogs/region_capacity_20261105.pt')
    for relative in [
        'scripts/data/prepare_region_capacity_data.py',
        'scripts/tuning/region_capacity_common.py',
        'scripts/tuning/run_region_capacity_study.py',
        'scripts/tuning/fit_region_etas.py',
        'scripts/tuning/fit_region_capacity_controls.py',
        'scripts/summarize/summarize_region_capacity.py',
        'scripts/summarize/audit_region_capacity_results.py',
        'scripts/summarize/audit_region_etas_results.py',
        'scripts/summarize/package_region_capacity.py',
        'tests/model/test_fast_region_etas.py',
    ]:add(ROOT/relative,relative)
    for name in ['environment.json','final_summary.json','final_junit.xml','old_checkpoint_replay.json','source_snapshot.json']:
        relative=REL/'reports/compatibility_20261105'/name;add(ROOT/relative,relative)
    manifest={name:dict(bytes=source.stat().st_size,sha256=sha(source)) for name,source in sorted(entries.items())}
    manifest_path=STORE/'bundle_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    archive=STORE/'region_capacity_20261105.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=4) as out:
        for name,source in sorted(entries.items()):out.add(source,arcname=name,recursive=False)
        out.add(manifest_path,arcname='bundle_manifest.json')
    metadata=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
        files=len(entries),manifest=str(manifest_path),
        excluded='Optimizer/resume states and unused intermediate checkpoints; originals remain on remote storage.')
    (STORE/'bundle.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
