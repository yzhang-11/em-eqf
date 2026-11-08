"""Portable project-layout backup of audited experimental outputs."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path('/root/autodl-tmp/em_eqf')
STORE=Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105')
REG=Path('experiments/revision/multiregion_pretraining')
CONTROL=Path('experiments/revision/controls')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit=ROOT/CONTROL/'reports/chuandian_multiregion_transfer_20261105/final_audit.json'
    assert json.loads(audit.read_text())['passed']
    assert json.loads((ROOT/CONTROL/'reports/chuandian_warmup_guard_20261105/final_audit.json').read_text())['passed']
    assert (ROOT/REG/'reports/mamba3_oracle_residual_20261105/results_zh.md').exists()
    files={}
    def add(source,relative):
        source=Path(source)
        if source.is_file():files[str(relative)]=source.resolve()
    for study,tag in [(REG,'mamba3_oracle_residual_20261105'),(CONTROL,'chuandian_multiregion_transfer_20261105'),
                      (CONTROL,'chuandian_warmup_guard_20261105')]:
        for branch in ['configs','shared/reports']:
            for source in (ROOT/study/branch/tag).rglob('*'):
                if source.suffix in {'.json','.yaml','.csv','.md','.xml','.log'}:add(source,source.relative_to(ROOT))
        for source in (ROOT/study/'runs'/tag).rglob('*'):
            if source.suffix in {'.json','.npz','.joblib'} or source.name=='best_model.pth':
                add(source,source.relative_to(ROOT))
    # Include the paired reference checkpoints and scores used in this report.
    for folder in (ROOT/REG/'runs/region_capacity_20261105').glob('*_small_long_seed_*'):
        for name in ['best_model.pth','training_complete.json','metrics_best_model_context_1024.json']:
            source=folder/name;add(source,source.relative_to(ROOT))
    for name in ['neural_results.json','summary.json','data.json','frozen_source.json']:
        relative=REG/'reports/region_capacity_20261105'/name;add(ROOT/relative,relative)
    relative=REG/'reports/region_capacity_20261105/etas'
    for source in (ROOT/relative).glob('*.json'):add(source,source.relative_to(ROOT))
    for branch in ['frozen_project','frozen_downstream','frozen_downstream_warmup_guard']:
        for source in (STORE/branch).rglob('*'):
            if source.suffix in {'.py','.yaml','.json'}:add(source,Path(branch)/source.relative_to(STORE/branch))
    for source in [STORE/'source_before.tar.gz',STORE/'oracle_wrapper_before.log',STORE/'compatibility.log',STORE/'compatibility.xml']:
        add(source,Path('code_audit')/source.name)
    add(STORE/'transfer_sources_before_20110212.pt','data/MultiRegion/catalogs/mamba3_transfer_sources_before_20110212_20261105.pt')
    add(STORE/'chuandian_transfer.pt','data/ChuanDian/catalogs/chuandian_multiregion_transfer_20261105.pt')
    add(ROOT/'data/MultiRegion/catalogs/region_capacity_20261105.pt','data/MultiRegion/catalogs/region_capacity_20261105.pt')
    add(ROOT/'data/ChuanDian/raw/chuandian_2021.csv','data/ChuanDian/raw/chuandian_2021.csv')
    for relative in [
        'scripts/tuning/mamba3_transfer_common.py','scripts/tuning/run_mamba3_transfer_pretraining.py',
        'scripts/tuning/run_chuandian_encoder_transfer.py','scripts/data/prepare_chuandian_encoder_transfer.py',
        'scripts/tuning/run_chuandian_warmup_guard.py',
        'scripts/summarize/summarize_mamba3_chuandian_transfer.py',
        'scripts/summarize/audit_mamba3_chuandian_transfer.py',
        'scripts/summarize/package_mamba3_chuandian_transfer.py',
        'tests/model/test_region_mixer_tpp.py']:
        add(ROOT/relative,relative)
    manifest={relative:dict(bytes=source.stat().st_size,sha256=sha(source)) for relative,source in sorted(files.items())}
    path=STORE/'bundle_manifest.json';path.write_text(json.dumps(manifest,indent=2))
    archive=STORE/'mamba3_chuandian_transfer_20261105.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=4) as out:
        for relative,source in sorted(files.items()):out.add(source,arcname=relative,recursive=False)
        out.add(path,arcname='bundle_manifest.json')
    result=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),
        excluded='Optimizer and resume states remain on the remote storage.')
    (STORE/'bundle.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
