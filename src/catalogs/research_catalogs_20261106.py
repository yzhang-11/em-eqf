"""New operational and research catalogs with explicit source-specific protocols."""
from pathlib import Path

from src.data import Catalog,default_catalogs_dir
from .regional_catalogs_20261105 import RegionalCatalog


class Catalog20261106(RegionalCatalog):
    schema='source-qualified-standard-20261106-v1'
    def __init__(self,root_dir=None,data_dir=None,catalog_file=None,**kwargs):
        ds=Path(data_dir) if data_dir is not None else default_catalogs_dir/self.dataset_name
        super().__init__(root_dir=root_dir or ds/'catalogs/standard_20261106',data_dir=ds,
            catalog_file=catalog_file or ds/'processed/standard_source_20261106.parquet',**kwargs)


@Catalog.register('USGS_Montana-Standard')
class USGSMontana(Catalog20261106):
    dataset_name=provider='USGS_Montana'
    default_mc=source_floor=2.0
    magnitude_resolution=0.0
    time_resolution_seconds=.001
    quality_protocol='regional-reviewed-strict-20261106-v1'


@Catalog.register('NORWAY_SW-Standard')
class NorwaySouthWest(Catalog20261106):
    dataset_name=provider='NORWAY_SW'
    default_start='2000-01-01'
    default_end='2024-01-01'
    default_validation='2014-01-01'
    default_test='2018-01-01'
    default_mc=source_floor=0.0
    magnitude_resolution=.1
    time_resolution_seconds=.001
    quality_protocol='research-relative-relocation-20261106-v1'


@Catalog.register('NORWAY_Bitdalsvatn-Standard')
class NorwayBitdalsvatn(Catalog20261106):
    dataset_name=provider='NORWAY_Bitdalsvatn'
    default_start='2020-07-01'
    default_end='2023-01-01'
    default_validation='2021-07-01'
    default_test='2022-01-01'
    default_mc=source_floor=0.0
    magnitude_resolution=0.0
    time_resolution_seconds=.000001
    quality_protocol='research-preferred-relocation-20261106-v1'
