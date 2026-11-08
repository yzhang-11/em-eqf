"""Independently versioned research catalogs released with a DOI."""
from src.data import Catalog
from .research_catalogs_20261106 import Catalog20261106


@Catalog.register("Greece_WestHellenic-Standard")
class GreeceWestHellenic(Catalog20261106):
    dataset_name = provider = "Greece_WestHellenic"
    default_start = "2023-12-01"
    default_validation = "2024-04-01"
    default_test = "2024-07-01"
    default_end = "2024-10-01"
    default_mc = source_floor = 1.3
    magnitude_resolution = .01
    time_resolution_seconds = .000001


@Catalog.register("Chile_Potin-Standard")
class ChilePotin(Catalog20261106):
    dataset_name = provider = "Chile_Potin"
    default_start = "2000-01-01"
    default_validation = "2014-01-01"
    default_test = "2017-01-01"
    default_end = "2020-07-01"
    default_mc = source_floor = 3.
    magnitude_resolution = .1
    time_resolution_seconds = .01


@Catalog.register("Chile_Atacama-Standard")
class ChileAtacama(Catalog20261106):
    dataset_name = provider = "Chile_Atacama"
    default_start = "2020-11-01"
    default_validation = "2022-07-01"
    default_test = "2023-04-01"
    default_end = "2024-02-15"
    default_mc = source_floor = 1.5
    magnitude_resolution = 0.
    time_resolution_seconds = .001


@Catalog.register("Greece_SantoriniSwarm-Standard")
class GreeceSantoriniSwarm(Catalog20261106):
    dataset_name = provider = "Greece_SantoriniSwarm"
    default_start = "2025-02-05"
    default_validation = "2025-02-12"
    default_test = "2025-02-19"
    default_end = "2025-03-01"
    default_mc = source_floor = 1.5
    magnitude_resolution = .01
    time_resolution_seconds = .000001
