import csv
import json
import sys

import pytest

from scripts.maintenance import index_experiments as index


def write_config(folder, name='config.yaml', seed=1):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(f'model: lstm\ndataset: Example\nseed: {seed}\n')


def test_nested_trials_are_indexed_once_and_ignore_log_directories(tmp_path):
    raw = tmp_path / 'data/Example/raw'
    raw.mkdir(parents=True)
    (raw / 'events.csv').write_text('t,Magnitude\n1,3\n')
    runs_dir = tmp_path / 'experiments/study/runs'
    parent = runs_dir / 'base'
    trial = parent / 'optuna_trials/trial_0000'
    write_config(parent)
    write_config(trial, 'config_input.yaml', seed=9)
    write_config(trial, seed=2)
    (trial / 'best_model_1.pth').write_bytes(b'weights')
    (trial / 'metrics_test_best_1.json').write_text('{}')
    write_config(parent / 'tensorboard/not_a_run', seed=99)
    (parent / 'optuna_trials/loop').symlink_to(runs_dir, target_is_directory=True)

    _, groups, runs, errors = index.build_index(tmp_path, tmp_path / 'index')
    assert not errors
    assert len(groups) == 1 and groups[0]['run_count'] == 2
    assert {r['seed'] for r in runs} == {1, 2}
    selected = next(r for r in runs if r['seed'] == 2)
    assert selected['config'].endswith('trial_0000/config.yaml')
    assert selected['checkpoints'].endswith('trial_0000/best_model_1.pth')
    assert selected['metrics'].endswith('trial_0000/metrics_test_best_1.json')
    assert {r['input_csv'] for r in runs} == {'data/Example/raw/events.csv'}


def test_study_with_optuna_trials_without_runs_directory_is_discovered(tmp_path):
    (tmp_path / 'data').mkdir()
    trial = tmp_path / 'experiments/search/optuna_trials/trial_0000'
    write_config(trial)
    _, groups, runs, errors = index.build_index(tmp_path, tmp_path / 'index')
    assert not errors
    assert [g['experiment'] for g in groups] == ['search']
    assert len(runs) == 1 and runs[0]['run'].endswith('optuna_trials/trial_0000')


def test_dataset_query_reads_saved_index_without_scanning_or_writing(tmp_path, monkeypatch, capsys):
    output = tmp_path / 'experiments/registry'
    output.mkdir(parents=True)
    path = output / 'runs.csv'
    rows = [{'experiment': 'study', 'dataset': 'Example', 'config': 'configs/run.yaml', 'input_csv': 'data/Example/raw/events.csv'}]
    index.write_csv(path, rows, list(rows[0]))
    original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns

    def fail(*args, **kwargs):
        raise AssertionError('A saved-index query must not scan the repository')

    monkeypatch.setattr(index, 'build_index', fail)
    monkeypatch.setattr(sys, 'argv', ['index_experiments.py', '--root', str(tmp_path), '--dataset', 'Example'])
    index.main()
    response = json.loads(capsys.readouterr().out)
    assert response['experiment'] == 'study' and response['runs'] == 1
    assert path.read_bytes() == original_bytes and path.stat().st_mtime_ns == original_mtime


def test_interrupted_refresh_does_not_publish_partial_csv(tmp_path, monkeypatch):
    path = tmp_path / 'runs.csv'
    path.write_text('previous complete index\n')

    def fail(writer, rows):
        writer.writerow(rows[0])
        raise RuntimeError('interrupted')

    monkeypatch.setattr(csv.DictWriter, 'writerows', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        index.write_csv(path, [{'seed': 1}], ['seed'])
    assert path.read_text() == 'previous complete index\n'
    assert not list(tmp_path.glob('.index-*'))
