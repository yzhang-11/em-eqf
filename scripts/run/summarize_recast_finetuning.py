"""Compare equally tuned local/fine RECAST and audit target-update efficiency."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

BASE = Path('/root/autodl-fs/em_eqf_training/unified_tpp_20261106')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curve(row, initial, reference, mode):
    if 'result' in row:
        history = row['result']['history']
    else:
        history = read(reference / 'gr' / row['target'] / f"s{row['seed']}" / 'runs' /
                       f"{row['target']}__{mode}__s{row['seed']}__lr0.0003/result.json")['history']
    return [(0, initial)] + [(r['step'], r['validation']['macro']['time']) for r in history]


def first_reached(points, threshold):
    return next((step for step, value in points if value <= threshold), None)


def main(root, baseline):
    assert read(root / 'recast_complete.json')['passed']
    baseline_audit = read(baseline / 'audit.json')
    assert baseline_audit['passed'] and baseline_audit['unconstrained_etas']
    assert sha(baseline / 'comparison.json') == baseline_audit['comparison_sha256']
    plan, prior_plan = read(root / 'plan.json'), read(baseline / 'plan.json')
    assert plan['mode'] == 'fine' and plan['targets'] == prior_plan['targets']
    assert sha(root / 'recast_driver.py') == plan['driver_sha256']
    for name, value in plan['source_files'].items():
        assert sha(root / 'project' / name) == value, name
    reference = Path(plan['reference'])
    assert sha(reference / 'gr/comparison.json') == plan['reference_sha256']
    local = read(baseline / 'recast_comparison.json')
    fine = read(root / 'recast_comparison.json')
    previous = read(baseline / 'comparison.json')
    initial = read(root / 'parent_validation.json')
    assert initial['passed'] and len(initial['rows']) == 18
    paired, speed = [], []
    for row in fine['rows']:
        target, seed = row['target'], row['seed']
        scratch = next(r for r in local['rows'] if (r['target'], r['seed']) == (target, seed))
        parent = next(r for r in initial['rows'] if (r['target'], r['seed']) == (target, seed))
        assert sha(Path(parent['parent_checkpoint'])) == parent['parent_sha256']
        assert row['events'] == scratch['events']
        if 'result' in row:
            assert sha(Path(row['checkpoint'])) == row['checkpoint_sha256']
            assert row['result']['parent_sha256'] == parent['parent_sha256']
            assert row['result']['draw_sha256'] == scratch['result']['draw_sha256']
            assert row['result']['initial_sha256'] == scratch['result']['initial_sha256']
            assert row['test_records'][0]['history_events'] == scratch['test_records'][0]['history_events']
        else:
            assert row['draw_sha256'] == scratch['draw_prefix_sha256']
        threshold = scratch['validation']
        local_curve = curve(scratch, parent['scratch_validation'], reference, 'local')
        fine_curve = curve(row, parent['validation'], reference, 'fine')
        for tolerance in [0., .01]:
            speed.append(dict(target=target, seed=seed, tolerance=tolerance,
                threshold=threshold + tolerance,
                local_updates=first_reached(local_curve, threshold + tolerance),
                fine_updates=first_reached(fine_curve, threshold + tolerance),
                source_run_updates=parent['source_updates'],
                source_checkpoint_step=parent['parent_best_step']))
        paired.append(dict(target=target, seed=seed,
            local_test=scratch['test_nll'], fine_test=row['test_nll'],
            fine_minus_local=row['test_nll'] - scratch['test_nll'],
            events=row['events'], parent_sha256=parent['parent_sha256']))
    summary, flat = {}, []
    for target in plan['targets']:
        old = previous['summary'][target]
        summary[target] = dict(local_tuned=local['summary'][target],
            fine_tuned=fine['summary'][target], zero=old['recast_zero'],
            fine800=old['recast_fine800'], etas=old['etas_unconstrained'])
        for model, value in summary[target].items():
            flat.append(dict(target=target, model=model, mean=value['mean'], sd=value['sd']))
    value = dict(passed=True, metric='physical-day time NLL/event; lower is better',
        seeds=[0, 1, 2], summary=summary, paired=paired,
        target_update_efficiency=speed, validation_tolerance=.01,
        efficiency_scope='First recorded validation crossing, every 100 updates; upstream pretraining cost reported separately; zero updates still uses observed history and auxiliary calibration',
        fine_selection=fine['selection'], local_selection=local['selection'],
        caveat='Equal target-update ceilings and LR search; pretraining adds upstream compute. No event-level significance claim.')
    (root / 'comparison.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    for name, rows in [('metrics.csv', flat), ('paired_deltas.csv', paired), ('target_update_efficiency.csv', speed)]:
        with (root / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit = dict(passed=True, pairs=18, same_target_data=True, parents_unchanged=True,
        paired_target_draws=True, matching_scored_events=True,
        comparison_sha256=sha(root / 'comparison.json'),
        local_reference_sha256=sha(baseline / 'comparison.json'),
        driver_sha256=sha(root / 'recast_driver.py'),
        analysis_files={name: sha(root / name) for name in
            ['audit_recast_parent_validation.py', 'summarize_recast_finetuning.py'] if (root / name).exists()},
        fine_mean_wins_local=sum(v['fine_tuned']['mean'] < v['local_tuned']['mean'] for v in summary.values()),
        fine_mean_wins_etas=sum(v['fine_tuned']['mean'] < v['etas']['mean'] for v in summary.values()))
    (root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    with tarfile.open(root / 'result_metadata.tar.gz', 'w:gz') as archive:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ['.json', '.csv', '.yaml', '.py', '.md', '.jsonl']:
                if not any(part in path.parts for part in ['project', '__pycache__']):
                    archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    print(json.dumps(dict(summary=summary, audit=audit)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE / 'benchmark_finetune_convergence_h4096_3seed')
    parser.add_argument('--baseline', type=Path, default=BASE / 'benchmark_convergence_h4096_3seed')
    args = parser.parse_args()
    main(args.root.resolve(), args.baseline.resolve())
