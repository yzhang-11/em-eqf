#!/usr/bin/env python3
"""Record data delivered by Git/LFS and inputs requiring a separate transfer."""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

GENERATED = {'catalogs', 'cache', '__pycache__', '.ipynb_checkpoints', 'deliverables', 'unified_tpp_20261106'}
WEIGHTS = {'.pt', '.pth', '.ckpt', '.pkl', '.pickle'}


def build_manifest(root):
    tracked = set(subprocess.check_output(
        ['git', 'ls-files', '-z', '--', 'data'], cwd=root).decode().split('\0')) - {'', 'data/external_sources.tar.gz'}
    datasets = json.loads((root / 'experiments/registry/datasets.json').read_text())
    inputs = {p for d in datasets for k in ('raw', 'processed') for p in d[k]}
    # Root aliases have no raw/processed children in some historical snapshots.
    aliases = {str(p.relative_to(root)) for p in (root / 'data').iterdir()
               if p.is_symlink()}
    hashes, files, links, private, missing = {}, {}, {}, [], []
    canonical = {str((root / name).resolve()): name for name in sorted(tracked)}

    def record(path, storage_alias=False, ancestors=frozenset()):
        relative = str(path.relative_to(root))
        if relative == 'data/CCL' or relative.startswith('data/CCL/'):
            private.append(relative)
            return
        if not path.exists():
            missing.append(relative)
            return
        target = path.resolve()
        if path.is_symlink():
            links[relative] = dict(target=str(target), directory=path.is_dir())
        if path.is_dir():
            if target in ancestors:
                raise ValueError(f'Circular data link: {path}')
            for child in sorted(path.iterdir()):
                if child.name in GENERATED:
                    continue
                if storage_alias and child.suffix in WEIGHTS:
                    continue
                record(child, storage_alias, ancestors | {target})
            return
        if relative in files:
            return
        if target not in hashes:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            hashes[target] = (path.stat().st_size, digest.hexdigest())
        size, digest = hashes[target]
        via = relative if relative in tracked else canonical.get(str(target))
        files[relative] = dict(path=relative, bytes=size, sha256=digest, source_path=str(target),
                               delivery='git' if via else 'external_bundle')
        if via and via != relative:
            files[relative]['via'] = via

    for relative in sorted(tracked | inputs | aliases):
        record(root / relative, storage_alias=relative in aliases)
    # Check actual data against LFS pointers, rather than hashing placeholders.
    lfs = subprocess.check_output(['git', 'lfs', 'ls-files', '--long'], cwd=root).decode()
    lfs_count = 0
    lfs_oids = {}
    for line in lfs.splitlines():
        oid, _, name = line.split(' ', 2)
        lfs_oids[name] = oid
        if name in files:
            if files[name]['sha256'] != oid:
                raise ValueError(f'LFS input is missing or changed: {name}')
            files[name]['delivery'] = 'git_lfs'
            lfs_count += 1
    for entry in files.values():
        if entry.get('via') in lfs_oids:
            if entry['sha256'] != lfs_oids[entry['via']]:
                raise ValueError(f'Changed data alias: {entry["path"]}')
            entry['delivery'] = 'git_lfs'
    return dict(schema_version=1, repository_root=str(root),
                external_bundle=dict(archive='data/external_sources.tar.gz', index='experiments/registry/external_sources.json'),
                storage_root=str(Path('/root/autodl-fs/em_eqf_catalogs').resolve()),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
                dataset_index='experiments/registry/datasets.json',
                scope='Tracked data, indexed raw/processed inputs, and source files in local root aliases; excludes generated caches and weights in root aliases.',
                counts=dict(tracked_data_files=len(tracked), data_lfs_files=lfs_count,
                            files_by_delivery=dict(Counter(v['delivery'] for v in files.values())),
                            private_excluded=len(set(private)), missing=len(set(missing))),
                private_excluded=sorted(set(private)), missing=sorted(set(missing)),
                links=[dict(path=p, **t) for p, t in sorted(links.items())],
                files=[files[p] for p in sorted(files)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = build_manifest(root)
    output = args.output or root / 'experiments/registry/data_release.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(manifest['counts'], indent=2))
    if manifest['missing']:
        raise SystemExit('Missing data inputs; inspect the manifest before publishing.')


if __name__ == '__main__':
    main()
