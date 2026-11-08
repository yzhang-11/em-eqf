#!/usr/bin/env python3
"""Retire obsolete caches and offload large historical runs after SHA-256 checks.

Default: print the plan. Pass --apply to perform it. Raw/processed inputs and
revision experiments are never selected. The archive manifest records all paths.
"""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(path):
    result = {}
    for folder, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            p = Path(folder) / name
            key = str(p.relative_to(path))
            if p.is_symlink():
                result[key] = {'link': os.readlink(p)}
            elif p.is_file():
                result[key] = {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
    return result


def archive_files(root, files, archive):
    """Verify the complete archive and source files before removing any file."""
    root, archive = root.resolve(), archive.resolve()
    files = [p for p in files if p.exists()]
    for p in files:
        require(not p.is_symlink() and p.resolve().is_relative_to(root), f'Unsafe cache path: {p}')
    before = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in files}
    if not files:
        return before
    require(not archive.exists(), f'Archive already exists: {archive}')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        require(set(tar.getnames()) == set(before), 'Archive member mismatch')
        for member in tar:
            h = hashlib.sha256()
            stream = tar.extractfile(member)
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                h.update(chunk)
            require(member.size == before[member.name]['bytes'], f'Archive size mismatch: {member.name}')
            require(h.hexdigest() == before[member.name]['sha256'], f'Archive hash mismatch: {member.name}')
    for p in files:
        require(file_hash(p) == before[str(p.relative_to(root))]['sha256'], f'Source changed: {p}')
    for p in files:
        p.unlink()
    return before


def offload(root, source, target):
    root, source, target = root.resolve(), source.absolute(), target.resolve()
    require(source.resolve().is_relative_to(root / 'experiments') and not source.is_symlink(), f'Unsafe source: {source}')
    require(not target.is_relative_to(root) and not target.exists(), f'Unsafe destination: {target}')
    before = snapshot(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, symlinks=True)
    require(snapshot(target) == before, f'Archive copy mismatch: {source}')
    require(snapshot(source) == before, f'Source changed during copy: {source}')
    # The copied data are independently verified before the local tree is removed.
    pending = source.with_name(source.name + '.offload-pending')
    require(not pending.exists(), f'Pending move already exists: {pending}')
    source.rename(pending)
    try:
        source.symlink_to(target, target_is_directory=True)
    except BaseException:
        pending.rename(source)
        raise
    shutil.rmtree(pending)
    return {'source': str(source.relative_to(root)), 'target': str(target), 'files': before}


def plan(root, min_bytes):
    old_cache_dirs = []
    for dataset in sorted((root / 'data').iterdir()):
        if not dataset.is_dir() or dataset.is_symlink():
            continue
        for name in ('classifier', 'regressor', 'lstm_regressor', 'rf_classifier', 'counter'):
            p = dataset / name
            if p.is_dir() and not p.is_symlink():
                old_cache_dirs.append(p)
    caches = [p for folder in old_cache_dirs for p in folder.rglob('*.pkl') if not p.is_symlink()]
    runs = []
    for family in ('mtpp', 'netas'):
        for folder, dirs, files in os.walk(root / 'experiments' / family, followlinks=False):
            for name in ('runs', 'profiles'):
                p = Path(folder) / name
                if name in dirs and not p.is_symlink():
                    size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
                    if size >= min_bytes:
                        runs.append((p, size))
            dirs[:] = [d for d in dirs if d not in {'runs', 'profiles', 'reports', 'tensorboard'} and not (Path(folder) / d).is_symlink()]
    return old_cache_dirs, caches, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--min-run-mb', type=int, default=64)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root, archive_root = args.root.resolve(), args.archive_root.resolve()
    require(not archive_root.is_relative_to(root), 'The archive must be outside the repository')
    cache_dirs, caches, runs = plan(root, args.min_run_mb * 1024 ** 2)
    summary = {'obsolete_cache_files': len(caches), 'cache_bytes': sum(p.stat().st_size for p in caches),
               'offload_dirs': len(runs), 'offload_bytes': sum(size for _, size in runs), 'apply': args.apply}
    print(json.dumps(summary), flush=True)
    if not args.apply:
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    journal_path = archive_root / 'manifest.json'
    require(not journal_path.exists(), 'Use a new archive directory for a new cleanup.')
    journal = {'plan': summary, 'disk_before': shutil.disk_usage(root)._asdict(), 'offloaded': []}
    write_json(journal_path, journal)
    journal['cache_archive'] = str(archive_root / 'obsolete_window_caches.tar.gz')
    journal['cache_files'] = archive_files(root, caches, Path(journal['cache_archive']))
    write_json(journal_path, journal)
    for folder in cache_dirs:
        if not any(folder.iterdir()):
            folder.rmdir()
    print('Verified and retired obsolete window caches', flush=True)
    for source, size in sorted(runs, key=lambda x: x[1], reverse=True):
        target = archive_root / source.relative_to(root)
        journal['offloaded'].append(offload(root, source, target))
        write_json(journal_path, journal)
        print('Verified and offloaded ' + str(source.relative_to(root)), flush=True)
    journal['disk_after'] = shutil.disk_usage(root)._asdict()
    write_json(journal_path, journal)
    print(json.dumps({'manifest': str(journal_path), 'disk_after': journal['disk_after']}), flush=True)


if __name__ == '__main__':
    main()
