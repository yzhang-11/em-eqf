"""Verify provenance and package the completed magnitude modeling experiments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT=Path('/root/autodl-tmp/em_eqf')
BASE=ROOT/'experiments/revision/magnitude_models'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def record(path):
    path=Path(path)
    return dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size)


def main():
    now=datetime.now(timezone.utc).isoformat()
    baseline=json.loads((ROOT/'experiments/revision/generation_baselines/frozen_manifest.json').read_text())
    frozen=[]
    for group in ['source_files','configs','raw_sources','analysis_files']:
        for entry in baseline[group]:
            assert digest(entry['path'])==entry['sha256'],entry['path']
            frozen.append(entry)
    original=json.loads((BASE/'plan.json').read_text())
    for entry in original['source_files']:
        assert digest(entry['path'])==entry['sha256'],entry['path']
    statuses=[]
    for path in sorted((BASE/'runs').glob('*/job_status.json')):
        value=json.loads(path.read_text());assert value['state']=='completed',path
        assert digest(value['checkpoint'])==value['checkpoint_sha256'],path
        statuses.append(value)
    original_statuses=[r for r in statuses if not r['name'].startswith('dynamics_')]
    dynamic_statuses=[r for r in statuses if r['name'].startswith('dynamics_')]
    assert len(original_statuses)==128,len(original_statuses)
    selection=json.loads((BASE/'selection.json').read_text())
    external=json.loads((BASE/'reports/external_2020_2024/plan.json').read_text())
    assert digest(BASE/'selection.json')==external['selection_sha256']
    assert json.loads((BASE/'reports/generation/status.json').read_text())['state']=='completed'
    sources=[ROOT/p for p in [
        'src/models/tpp/magnitude_density.py','src/models/tpp/magnitude_model.py',
        'src/models/tpp/magnitude_state.py','src/models/tpp/magnitude_gated.py',
        'scripts/data/prepare_magnitude_features.py','scripts/run/run_magnitude_models.py',
        'scripts/run/run_magnitude_states.py','scripts/run/run_magnitude_controls.py',
        'scripts/run/run_magnitude_gated.py','scripts/run/select_magnitude_memory.py',
        'scripts/analyze/compare_magnitude_models.py','scripts/analyze/calibrate_magnitude_models.py',
        'scripts/analyze/evaluate_magnitude_external.py','scripts/analyze/evaluate_magnitude_generation.py',
        'scripts/diagnostics/validate_magnitude_models.py','scripts/summarize/export_magnitude_models.py',
        'tests/distribution/test_magnitude_density.py','tests/model/test_magnitude_state.py',
        'tests/model/test_magnitude_model_history.py','tests/model/test_magnitude_gated.py']]
    sources += [ROOT/p for p in [
        'src/models/tpp/magnitude_dynamics.py','src/models/tpp/magnitude_dynamics_model.py',
        'scripts/run/run_magnitude_dynamics.py','scripts/analyze/compare_magnitude_dynamics.py',
        'scripts/analyze/evaluate_magnitude_dynamics_external.py',
        'scripts/diagnostics/validate_magnitude_dynamics.py',
        'tests/model/test_magnitude_dynamics.py','tests/model/test_magnitude_dynamics_model.py']]
    selected=[];bases=[]
    for region in ['swc','scsn']:
        recipe=selection['recipes'][region]['selected']
        for variant in ['main','sensitivity']:
            for seed in range(5):
                name=f"confirm_{region}_{variant}_{recipe['family']}{recipe['components']}_{recipe['features']}_seed_{seed}"
                selected.append(BASE/'runs'/name)
                bases.append(ROOT/'experiments/revision/main/runs'/f'generation_{region}_{variant}_seed_{seed}'/'best_model_1.pth')
    inputs=list((BASE/'features').glob('*'))+list((BASE/'configs').glob('*.json'))
    inputs+=[BASE/'plan.json',BASE/'selection.json']+bases+sources
    inputs+=list((ROOT/'data/SCEDC/raw/scsn_annual_snapshot_20261104').glob('*.catalog'))
    inputs.append(ROOT/'data/SCEDC/processed/scsn_eq_2020_2024_snapshot_20261104.csv')
    dynamic_checkpoints=[]
    dynamic_selection=BASE/'reports/dynamics/selection.json'
    if dynamic_selection.exists():
        for row in json.loads(dynamic_selection.read_text())['best']:
            if row['family']=='gr_free':dynamic_checkpoints.append(Path(row['checkpoint']))
        inputs.append(dynamic_selection)
    (BASE/'manifest.json').write_text(json.dumps(dict(created_utc=now,
        completed_fits=len(statuses),selected_head_checkpoints=[record(q/'best_head.pth') for q in selected],
        original_density_fits=len(original_statuses),dynamic_optimizations=len(dynamic_statuses),
        selected_dynamic_checkpoints=[record(q) for q in dynamic_checkpoints],
        current_inputs=[record(q) for q in sorted(set(inputs)) if q.is_file()],
        original_baseline_verified=frozen,
        scope='input/output provenance; does not change original per-stage frozen plans'),indent=2)+'\n')
    (BASE/'reports/conclusions.json').write_text(json.dumps(dict(created_utc=now,
        completed_fits=len(statuses),original_density_fits=128,dynamic_optimizations=len(dynamic_statuses),
        distribution_tests_passed=27,
        latest_temporal_candidate='GR-GAS; three likelihood-trained coefficients; lower adjacent jumps, no hard calendar-rate bound',
        latest_result_directory='reports/dynamics',
        first_stage_conditional_candidate='SCSN: two-component Kumaraswamy, context_history, NLL only, frozen original encoder/time head',
        first_stage_original_test_gain_nats_event=.012482,first_stage_external_2020_2024_gain_nats_event=.003653,
        latest_caveats=['GAS initialized fits are stable; arbitrary learned-coefficient perturbation is not invariant',
                        'GAS reduces adjacent jumps, not full-period b standard deviation',
                        'Original test intervals cross zero; 2025 higher Mc does not improve',
                        'Full-generation results are mixed; defaults unchanged'],
        first_stage_caveats=['Original-test block-1 CI crosses zero; no multiplicity adjustment',
                 'External-period intervals cross zero',
                 'Higher Mc original test does not improve',
                 'Probability variation increases; not a smoothing solution',
                 'Single-fit generation has worse count coverage; full replacement unsupported',
                 'SWC has no reliable improvement'],
        defaults_changed=False,rejected_on_validation=['Markov hidden states','time memory tau>0','shared global shapes with neural gates'],
        first_stage_generation=dict(fits_per_region=1,draws_per_window=128,windows=116,new_trajectories=14848)),indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='magnitude_delivery_') as folder:
        stage=Path(folder)/'magnitude_models_20261104';stage.mkdir()
        def copy(path,target):
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        for name in ['README.md','manifest.json','plan.json','selection.json']:
            copy(BASE/name,stage/name)
        for path in (BASE/'configs').glob('*.json'):copy(path,stage/'configs'/path.name)
        for path in (BASE/'shared/reports').rglob('*'):
            if path.is_file() and 'failed_attempt' not in str(path):copy(path,stage/path.relative_to(BASE))
        for path in (BASE/'features').glob('*.json'):copy(path,stage/'features'/path.name)
        dynamic_runs=[Path(r['checkpoint']).parent for r in dynamic_statuses]
        for run in selected+dynamic_runs:
            for path in run.iterdir():
                if path.is_file():copy(path,stage/path.relative_to(BASE))
        for path in sources:copy(path,stage/'source'/path.relative_to(ROOT))
        records=[dict(path=str(q.relative_to(stage)),sha256=digest(q),bytes=q.stat().st_size)
                 for q in sorted(stage.rglob('*')) if q.is_file()]
        (stage/'delivery_manifest.json').write_text(json.dumps(dict(created_utc=now,files=records,
            omitted='feature arrays, original base checkpoints, unselected heads and preliminary failed external attempts remain on server'),indent=2)+'\n')
        archive=Path('/tmp/magnitude_models_20261104.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:tar.add(stage,arcname=stage.name)
        print(json.dumps(dict(archive=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,
            delivered_files=len(records)+1,original_frozen_records_verified=len(frozen),completed_fits=len(statuses)),indent=2))


if __name__=='__main__':main()
