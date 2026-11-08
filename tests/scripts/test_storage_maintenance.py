import tarfile

import pytest

from scripts.maintenance import clean_storage as storage
from scripts.maintenance import reorganize_data_layout as layout


def test_archive_preserves_verified_copy_before_removing_cache(tmp_path):
    root = tmp_path / 'repo'
    cache = root / 'data/Example/classifier/old.pkl'
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'cached-windows')
    raw = root / 'data/Example/raw/events.csv'
    raw.parent.mkdir()
    raw.write_text('original source')
    archive = tmp_path / 'archive/caches.tar.gz'
    records = storage.archive_files(root, [cache], archive)
    assert not cache.exists()
    assert raw.read_text() == 'original source'
    with tarfile.open(archive) as tar:
        assert tar.extractfile('data/Example/classifier/old.pkl').read() == b'cached-windows'
    assert records['data/Example/classifier/old.pkl']['bytes'] == 14


def test_offload_verifies_bytes_and_keeps_one_access_path(tmp_path):
    root = tmp_path / 'repo'
    source = root / 'experiments/study/runs'
    source.mkdir(parents=True)
    (source / 'model.pth').write_bytes(b'weights')
    target = tmp_path / 'archive/study/runs'
    record = storage.offload(root, source, target)
    assert source.is_symlink()
    assert (source / 'model.pth').read_bytes() == b'weights'
    assert (target / 'model.pth').read_bytes() == b'weights'
    assert record['files']['model.pth']['sha256'] == storage.file_hash(target / 'model.pth')


def test_mismatched_offload_never_removes_source(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    source = root / 'experiments/study/runs'
    source.mkdir(parents=True)
    (source / 'model.pth').write_bytes(b'weights')
    target = tmp_path / 'archive/study/runs'
    original = storage.snapshot
    monkeypatch.setattr(storage, 'snapshot', lambda p: {'corrupt': True} if p == target else original(p))
    with pytest.raises(RuntimeError, match='copy mismatch'):
        storage.offload(root, source, target)
    assert not source.is_symlink()
    assert (source / 'model.pth').read_bytes() == b'weights'


def test_data_layout_dry_run_has_no_filesystem_side_effects(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'events.csv').write_text('source')
    before = sorted(str(p.relative_to(dataset)) for p in dataset.rglob('*'))
    layout.reorganize_dataset(dataset, layout.Stats(), dry_run=True)
    assert sorted(str(p.relative_to(dataset)) for p in dataset.rglob('*')) == before


def test_data_layout_leaves_caches_and_reports_out_of_raw(tmp_path):
    for name in ('cache', 'reports', 'analysis_plots'):
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'artifact.txt').write_text(name)
    layout.reorganize_dataset(tmp_path, layout.Stats(), dry_run=False)
    for name in ('cache', 'reports', 'analysis_plots'):
        assert (tmp_path / name / 'artifact.txt').read_text() == name
        assert not (tmp_path / 'raw' / name).exists()
