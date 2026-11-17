"""Exercise data recovery, immutable existing inputs and archive path validation."""
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/maintenance'))
from transfer_external_data import bundle, restore, sha256


class DataReleaseTransferTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root, self.store = base / 'repo', base / 'store'
        (self.root / 'experiments/registry').mkdir(parents=True)
        self.store.mkdir()
        (self.store / 'source.csv').write_text('time,magnitude\n1,2\n')
        (self.store / 'catalogs').mkdir()
        (self.store / 'catalogs/generated.pt').write_bytes(b'cache')
        (self.store / 'model.pt').write_bytes(b'weights')
        (self.store / 'CCL').mkdir()
        (self.store / 'CCL/private.csv').write_text('private data')
        self.archive = self.root / 'data/external_sources.tar.gz'
        self.index = self.root / 'experiments/registry/external_sources.json'
        self.audit_path = self.root / 'experiments/registry/data_release.json'
        self.audit = dict(storage_root=str(self.store), repository_root=str(self.root),
                          files=[dict(path='data/Test/processed/source.csv', delivery='external_bundle',
                                      source_path=str(self.store / 'source.csv'))],
                          links=[dict(path='data/Test/processed/source.csv', target=str(self.store / 'source.csv'), directory=False)])
        self.audit_path.write_text(json.dumps(self.audit))

    def test_roundtrip_and_conflict(self):
        bundle(self.root, self.store, self.archive, self.index)
        metadata = json.loads(self.index.read_text())
        self.assertEqual([e['path'] for e in metadata['files']], ['source.csv'])
        destination = self.store.parent / 'restored_store'
        restore(self.root, destination, self.archive, self.index)
        self.assertEqual((self.root / 'data/Test/processed/source.csv').read_text(), (self.store / 'source.csv').read_text())
        restore(self.root, destination, self.archive, self.index)
        (destination / 'source.csv').write_text('different data')
        with self.assertRaises(FileExistsError):
            restore(self.root, destination, self.archive, self.index)
        self.assertEqual((destination / 'source.csv').read_text(), 'different data')

    def test_uncovered_input_refuses_bundle(self):
        self.audit['files'][0]['source_path'] = str(self.store.parent / 'missing.csv')
        self.audit_path.write_text(json.dumps(self.audit))
        with self.assertRaises(ValueError):
            bundle(self.root, self.store, self.archive, self.index)
        self.assertFalse(self.archive.exists())

    def test_path_traversal_refused(self):
        self.archive.parent.mkdir()
        payload = b'escape'
        with tarfile.open(self.archive, 'w:gz') as tar:
            member = tarfile.TarInfo('../escaped.txt')
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
        self.index.write_text(json.dumps(dict(bytes=self.archive.stat().st_size, sha256=sha256(self.archive),
                                             files=[dict(path='../escaped.txt', bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())])))
        with self.assertRaises(ValueError):
            restore(self.root, self.store.parent / 'restored_store', self.archive, self.index)
        self.assertFalse((self.store.parent / 'escaped.txt').exists())

    def test_corrupted_archive_refused(self):
        bundle(self.root, self.store, self.archive, self.index)
        with self.archive.open('ab') as stream:
            stream.write(b'corruption')
        destination = self.store.parent / 'restored_store'
        with self.assertRaises(ValueError):
            restore(self.root, destination, self.archive, self.index)
        self.assertFalse(destination.exists())
