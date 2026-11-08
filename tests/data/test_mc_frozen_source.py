import json

import numpy as np
import pandas as pd

from src.catalogs.scedc import SCEDC


def test_local_catalog_honors_mc_and_changes_cache_when_source_changes(tmp_path, monkeypatch):
    def unexpected_network(**kwargs):
        raise AssertionError('Frozen local catalog must not download current data')

    monkeypatch.setattr('src.catalogs.scedc.requests.get', unexpected_network)
    source = tmp_path / 'events.csv'
    frame = pd.DataFrame({'ts': ['1981-01-02', '1981-01-03', '1981-01-04'],
                          'magnitude': [2.49, 2.5, 3.]})
    frame.to_csv(source, index=False)
    catalog = SCEDC(root_dir=tmp_path / 'cache', catalog_file=source, mag_completeness=2.5,
                    train_start_ts='1981-01-01', val_start_ts='1981-01-03', test_start_ts='1981-01-04')
    np.testing.assert_allclose(catalog.full_sequence.mag, [2.5, 3.])
    assert catalog.metadata['mag_completeness'] == 2.5
    assert json.loads((catalog.root_dir / 'catalog_cfg.json').read_text())['source_sha256']
    frame.loc[0, 'magnitude'] = 2.6
    frame.to_csv(source, index=False)
    revised = SCEDC(root_dir=tmp_path / 'cache', catalog_file=source, mag_completeness=2.5,
                    train_start_ts='1981-01-01', val_start_ts='1981-01-03', test_start_ts='1981-01-04')
    assert revised.root_dir != catalog.root_dir
    assert len(revised.full_sequence.mag) == 3
