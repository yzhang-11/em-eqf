"""Additional reviewed regional catalogs using the existing standard adapter."""
from src.data import Catalog
from .external_catalog import ExternalCatalog


class RegionalCatalog(ExternalCatalog):
    default_start='2014-01-01'
    default_validation='2019-01-01'
    default_test='2022-01-01'

    def __init__(self,root_dir=None,data_dir=None,catalog_file=None,mag_completeness=None,
                 normalize=True,start_ts=None,end_ts=None,train_start_ts=None,
                 val_start_ts=None,test_start_ts=None):
        super().__init__(root_dir=root_dir,data_dir=data_dir,catalog_file=catalog_file,
            mag_completeness=mag_completeness,normalize=normalize,start_ts=start_ts,end_ts=end_ts,
            train_start_ts=train_start_ts,val_start_ts=val_start_ts or self.default_validation,
            test_start_ts=test_start_ts or self.default_test)


@Catalog.register('USGS_Alaska-Standard')
class USGSAlaska(RegionalCatalog):
    dataset_name = provider = 'USGS_Alaska'
    default_mc = source_floor = 3.0
    magnitude_resolution = 0.0  # Preferred magnitude methods/reporting grids vary.
    time_resolution_seconds = .001


@Catalog.register('USGS_Hawaii-Standard')
class USGSHawaii(RegionalCatalog):
    dataset_name = provider = 'USGS_Hawaii'
    default_mc = source_floor = 2.5
    magnitude_resolution = 0.0
    time_resolution_seconds = .01


@Catalog.register('USGS_PuertoRico-Standard')
class USGSPuertoRico(RegionalCatalog):
    dataset_name = provider = 'USGS_PuertoRico'
    default_mc = source_floor = 2.5
    magnitude_resolution = 0.0
    time_resolution_seconds = .01


@Catalog.register('SED_Switzerland-Standard')
class SEDSwitzerland(RegionalCatalog):
    dataset_name = provider = 'SED_Switzerland'
    default_mc = source_floor = 2.0
    magnitude_resolution = 0.0
    time_resolution_seconds = .000001

