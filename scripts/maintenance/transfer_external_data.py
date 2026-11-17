#!/usr/bin/env python3
"""Bundle or restore external source snapshots without caches or private CCL data."""
import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

from index_data_release import GENERATED, WEIGHTS


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bundle(root, store, archive, index):
    files = [p for p in sorted(store.rglob('*')) if p.is_file()
             and not any(part in GENERATED | {'CCL', '.git'} for part in p.relative_to(store).parts)
             and p.suffix not in WEIGHTS]
    if not files:
        raise ValueError(f'No source files at {store}')
    known = {p.resolve() for p in files}
    audit = json.loads((root / 'experiments/registry/data_release.json').read_text())
    uncovered = [f['path'] for f in audit['files'] if f['delivery'] == 'external_bundle'
                 and Path(f['source_path']) not in known]
    if uncovered:
        raise ValueError(f'Inputs not covered by the source bundle: {uncovered[:10]}')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + '.tmp')
    entries = []
    try:
        with tarfile.open(temporary, 'w:gz', compresslevel=1, dereference=True) as tar:
            for source in files:
                name = source.relative_to(store).as_posix()
                entry = dict(path=name, bytes=source.stat().st_size, sha256=sha256(source))
                tar.add(source, arcname=name, recursive=False)
                entries.append(entry)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = dict(schema_version=1, storage_root=str(store), archive=str(archive.relative_to(root)),
                    bytes=archive.stat().st_size, sha256=sha256(archive), files=entries)
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'files'}, indent=2), flush=True)
    print(f'Bundled {len(entries)} source files; all separately delivered inputs covered.', flush=True)


def restore(root, store, archive, index):
    metadata = json.loads(index.read_text())
    if archive.stat().st_size != metadata['bytes'] or sha256(archive) != metadata['sha256']:
        raise ValueError('Archive checksum mismatch or LFS pointer not downloaded')
    expected = {entry['path']: entry for entry in metadata['files']}
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            entry = expected.get(member.name)
            destination = store / member.name
            if not entry or member.name in seen or not member.isfile() or not destination.resolve().is_relative_to(store):
                raise ValueError(f'Unexpected or unsafe archive member: {member.name}')
            if member.size != entry['bytes']:
                raise ValueError(f'Archive member size mismatch: {member.name}')
            seen.add(member.name)
            if destination.exists():
                if not destination.is_file() or sha256(destination) != entry['sha256']:
                    raise FileExistsError(f'Refusing to overwrite different data: {destination}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                digest = hashlib.sha256()
                try:
                    source = tar.extractfile(member)
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        stream.write(block)
                        digest.update(block)
                    stream.close()
                    if digest.hexdigest() != entry['sha256']:
                        raise ValueError(f'Archive member checksum mismatch: {member.name}')
                    if destination.exists():
                        raise FileExistsError(destination)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
    if seen != expected.keys():
        raise ValueError('Archive does not contain all expected files')
    audit = json.loads((root / 'experiments/registry/data_release.json').read_text())
    old_store, old_root = Path(audit['storage_root']), Path(audit['repository_root'])
    for link in audit['links']:
        path, original = root / link['path'], Path(link['target'])
        if not path.resolve().is_relative_to(root) and not path.is_symlink():
            raise ValueError(f'Unsafe link location: {path}')
        if original.is_relative_to(old_store):
            target = store / original.relative_to(old_store)
        elif original.is_relative_to(old_root):
            target = root / original.relative_to(old_root)
        else:
            raise ValueError(f'Unknown link target: {original}')
        if link['directory']:
            target.mkdir(parents=True, exist_ok=True)
        elif not target.is_file():
            raise FileNotFoundError(target)
        if path.is_symlink() and path.resolve() == target.resolve():
            continue
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing to replace existing data: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=link['directory'])
    print(f'Restored/verified {len(seen)} source files and {len(audit["links"])} data links.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('bundle', 'restore'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--store-root', type=Path, default=Path('/root/autodl-fs/em_eqf_catalogs'))
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    root, store = args.root.resolve(), args.store_root.resolve()
    archive = args.archive or root / 'data/external_sources.tar.gz'
    index = root / 'experiments/registry/external_sources.json'
    (bundle if args.action == 'bundle' else restore)(root, store, archive, index)


if __name__ == '__main__':
    main()
