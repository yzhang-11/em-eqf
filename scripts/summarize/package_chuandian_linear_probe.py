"""Archive the new study without altering previous experiment artifacts."""
import hashlib
import argparse
import json
from pathlib import Path
import tarfile

ROOT = Path('/root/autodl-tmp/em_eqf')
STORE = Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pool', choices=['last', 'mean'], default='last')
args = parser.parse_args()
suffix = '_mean' if args.pool == 'mean' else ''
TAG = f'chuandian_linear_probe{suffix}_20261105'
WORK = STORE / f'linear_probe{suffix}'
REPORTS = ROOT / 'experiments/revision/controls/reports' / TAG


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


assert json.loads((REPORTS / 'final_audit.json').read_text())['passed']
items = [
    (WORK, WORK.relative_to(STORE)),
    (REPORTS, REPORTS.relative_to(ROOT)),
    (ROOT / 'experiments/revision/controls/configs' / TAG, Path('experiments/revision/controls/configs') / TAG),
]
for name in ['tuning/run_chuandian_linear_probe.py', 'summarize/summarize_chuandian_linear_probe.py', 'summarize/package_chuandian_linear_probe.py']:
    items.append((ROOT / 'scripts' / name, Path('scripts') / name))
comparison = ROOT / 'scripts/summarize/compare_chuandian_readouts.py'
if args.pool == 'mean' and comparison.is_file():
    items.append((comparison, comparison.relative_to(ROOT)))
manifest = {}
for source, relative in items:
    paths = source.rglob('*') if source.is_dir() else [source]
    for path in paths:
        if path.is_file() and not path.is_symlink() and path.name != 'bundle_manifest.json':
            archive_path = relative / path.relative_to(source) if source.is_dir() else relative
            manifest[str(archive_path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
(REPORTS / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2))
archive = STORE / f'{TAG}.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
    for source, relative in items:
        tar.add(source, arcname=str(relative))
meta = dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), regular_files=len(manifest))
(STORE / f'{TAG}_bundle.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta))
