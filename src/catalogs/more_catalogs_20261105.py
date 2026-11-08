"""Further reviewed catalogs; existing adapters and caches remain unchanged."""
from src.data import Catalog
from .regional_catalogs_20261105 import RegionalCatalog


@Catalog.register('USGS_PNW-Standard')
class USGSPacificNorthwest(RegionalCatalog):
    dataset_name=provider='USGS_PNW'
    default_mc=source_floor=2.0
    magnitude_resolution=0.0
    time_resolution_seconds=.01


@Catalog.register('USGS_Utah-Standard')
class USGSUtah(RegionalCatalog):
    dataset_name=provider='USGS_Utah'
    default_mc=source_floor=2.0
    magnitude_resolution=0.0
    time_resolution_seconds=.01


@Catalog.register('USGS_NewMadrid-Standard')
class USGSNewMadrid(RegionalCatalog):
    dataset_name=provider='USGS_NewMadrid'
    default_mc=source_floor=2.0
    magnitude_resolution=0.0
    time_resolution_seconds=.01


@Catalog.register('NOA_Greece-Standard')
class NOAGreece(RegionalCatalog):
    dataset_name=provider='NOA_Greece'
    # Earlier snapshots lack required event-type/horizontal-quality metadata.
    # Keep them in the source audit, outside the model observation window.
    default_start='2018-01-01'
    default_mc=source_floor=2.5
    magnitude_resolution=0.0
    time_resolution_seconds=.000001


@Catalog.register('ISOR_Hengill-Standard')
class ISORHengill(RegionalCatalog):
    dataset_name=provider='ISOR_Hengill'
    default_start='2018-12-01'
    default_end='2021-02-01'
    default_validation='2020-01-01'
    default_test='2020-07-01'
    default_mc=source_floor=1.0
    magnitude_resolution=0.0
    time_resolution_seconds=.000001
