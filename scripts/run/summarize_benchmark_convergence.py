"""Audit and summarize completed RECAST/ETAS convergence experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    for name in ['recast', 'etas']:
        assert read(root / f'{name}_complete.json')['passed']
    plan = read(root / 'plan.json')
    reference = Path(plan['reference'])
    previous = read(reference / 'final_comparison.json')
    recast = read(root / 'recast_comparison.json')
    etas = read(root / 'etas_comparison.json')
    assert etas['enforce_p_gt_one'] is False and etas['enforce_subcritical'] is False
    summary, flat = {}, []
    for target in plan['targets']:
        records = {}
        for seed in [0, 1, 2]:
            neural = next(r for r in recast['rows'] if (r['target'], r['seed']) == (target, seed))
            classical = next(r for r in etas['rows'] if (r['target'], r['seed']) == (target, seed))
            assert neural['events'] == classical['test']['events']
            cfg = classical['config']
            assert cfg['etas_enforce_p_gt_one'] is False and cfg['etas_enforce_subcritical'] is False
            assert not any(k in cfg for k in ['etas_max_branching_ratio', 'etas_min_omori_p'])
            assert sha(Path(classical['checkpoint'])) == classical['checkpoint_sha256']
            if 'checkpoint' in neural:
                assert sha(Path(neural['checkpoint'])) == neural['checkpoint_sha256']
                assert neural['test_records'][0]['history_events'] == classical['test']['history_events']
            records[str(seed)] = dict(recast_events=neural['events'], etas_test=classical['test'],
                                     etas_parameters=classical['parameters'])
        old = previous['summary'][target]
        row = dict(recast_local800=old['gr']['local'], recast_local_tuned=recast['summary'][target],
                   recast_zero=old['gr']['zero'], recast_fine800=old['gr']['fine'],
                   etas_unconstrained=etas['summary'][target],
                   etas_historical_constrained=old['validation_selected_etas'], seed_audit=records)
        summary[target] = row
        for name, value in row.items():
            if name != 'seed_audit':
                flat.append(dict(target=target, model=name, mean=value['mean'], sd=value['sd']))
    result = dict(passed=True, metric='physical-day time NLL per event; lower is better',
                  history=4096, train_events=2048, seeds=[0, 1, 2], summary=summary,
                  reference=str(reference / 'final_comparison.json'),
                  recast_selection=recast['selection'], etas_selection=etas['selection'],
                  caveat='Long RECAST budgets match update count, not FLOPs; cross-region models are the existing frozen checkpoints')
    (root / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (root / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['target', 'model', 'mean', 'sd'])
        writer.writeheader()
        writer.writerows(flat)
    audit = dict(passed=True, comparison_sha256=sha(root / 'comparison.json'),
                 original_reference_unchanged=sha(reference / 'gr/comparison.json') == plan['reference_sha256'],
                 matching_event_counts_and_history=True, unconstrained_etas=True,
                 recast_driver_sha256=sha(root / 'recast_driver.py'),
                 etas_driver_sha256=sha(root / 'etas_driver.py'))
    assert audit['original_reference_unchanged']
    (root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    excluded = {'best.pt', 'best.tmp', 'best_model.pth'}
    with tarfile.open(root / 'result_metadata.tar.gz', 'w:gz') as archive:
        for path in root.rglob('*'):
            if path.is_file() and path.name not in excluded and path.suffix in ['.json', '.yaml', '.csv', '.py', '.md', '.jsonl']:
                if any(part in path.parts for part in ['project', '__pycache__']):
                    continue
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    print(json.dumps({t: {k: v['mean'] for k, v in row.items() if k != 'seed_audit'}
                      for t, row in summary.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-fs/em_eqf_training/unified_tpp_20261106/benchmark_convergence_h4096_3seed'))
    main(parser.parse_args().root.resolve())
