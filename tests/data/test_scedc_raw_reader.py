from types import SimpleNamespace

import pandas as pd

from src.catalogs import scedc


def test_scedc_download_keeps_first_event_after_commented_header(monkeypatch, tmp_path):
    text = ('#YYY/MM/DD HH:mm:SS.ss ET GT MAG M LAT LON DEPTH Q EVID NPH NGRM\n'
            '1981/01/01 01:49:29.23 eq l 2.27 h 33.724 -118.832 0.6 A 3301561 30 161\n'
            '1981/01/01 04:13:55.64 eq l 2.26 h 33.249 -115.967 1.0 A 3301565 46 161\n')
    monkeypatch.setattr(scedc.requests, 'get', lambda **kwargs: SimpleNamespace(content=text.encode()))
    catalog = object.__new__(scedc.SCEDC)
    catalog.catalog_file = None
    catalog.root_dir = tmp_path
    catalog.metadata = {'start_ts': pd.Timestamp('1981-01-01'),
                        'end_ts': pd.Timestamp('1982-01-01'), 'mag_completeness': 2.}
    catalog.generate_catalog()
    frame = pd.read_csv(tmp_path / 'catalog.csv')
    assert frame.EVID.tolist() == [3301561, 3301565]
    assert frame.magnitude.tolist() == [2.27, 2.26]
