"""Independent point-process views built after the 2026-11-06 integrity audit."""
from pathlib import Path

from src.data import Catalog, default_catalogs_dir
from .external_catalog import utc_naive
from .regional_catalogs_20261105 import RegionalCatalog

STORE = Path("/root/autodl-fs/em_eqf_catalogs/remediation_20261106")


class RemediatedCatalog(RegionalCatalog):
    schema = "source-qualified-remediation-20261106-v1"
    source_file = None

    def __init__(self, root_dir=None, data_dir=None, catalog_file=None, **kwargs):
        default_file = self.source_file
        if isinstance(default_file, str):
            default_file = default_catalogs_dir / default_file
        if utc_naive(kwargs.get("start_ts") or self.default_start) < utc_naive(self.default_start):
            raise ValueError(f"{self.dataset_name} starts on {self.default_start}; use the original catalog for earlier dates")
        if utc_naive(kwargs.get("end_ts") or self.default_end) > utc_naive(self.default_end):
            raise ValueError(f"{self.dataset_name} ends on {self.default_end}")
        super().__init__(
            root_dir=root_dir or STORE / self.dataset_name / "catalogs",
            data_dir=data_dir,
            catalog_file=catalog_file or default_file,
            **kwargs,
        )


@Catalog.register("NCEDC_Repaired20261106-Standard")
class NCEDCRepaired(RemediatedCatalog):
    dataset_name = "NCEDC_Repaired20261106"
    provider = "NCEDC"
    source_file = STORE / "NCEDC/processed/quality_screened.parquet"
    default_start = "1985-01-01"
    default_validation = "2014-01-01"
    default_test = "2016-01-01"
    default_mc = source_floor = 2.5
    magnitude_resolution = 0.01
    time_resolution_seconds = 0.001


@Catalog.register("USGS_MontanaComCat20261106-Standard")
class MontanaComCat(RemediatedCatalog):
    dataset_name = "USGS_MontanaComCat20261106"
    provider = "USGS_MontanaComCat"
    source_file = STORE / "USGS_MontanaComCat/processed/quality_screened_20261105.parquet"
    default_start = "2016-01-01"
    default_mc = source_floor = 2.0
    magnitude_resolution = 0.0
    time_resolution_seconds = 0.001


@Catalog.register("Taiwan_CWA1993Plus20261106-Standard")
class TaiwanCWA1993Plus(RemediatedCatalog):
    dataset_name = "Taiwan_CWA1993Plus20261106"
    provider = "Taiwan_CWA"
    source_file = "Taiwan_CWA/processed/standard_source_20261106.parquet"
    default_start = "1993-01-01"
    default_mc = source_floor = 3.0
    magnitude_resolution = 0.1
    time_resolution_seconds = 0.01


@Catalog.register("Chile_Potin2001NovPlus20261106-Standard")
class ChilePotin2001NovPlus(RemediatedCatalog):
    dataset_name = "Chile_Potin2001NovPlus20261106"
    provider = "Chile_Potin"
    source_file = "Chile_Potin/processed/standard_source_20261106.parquet"
    default_start = "2001-11-01"
    default_validation = "2014-01-01"
    default_test = "2017-01-01"
    default_end = "2020-07-01"
    default_mc = source_floor = 3.0
    magnitude_resolution = 0.1
    time_resolution_seconds = 0.01


class EAFZView(RemediatedCatalog):
    provider = "Turkiye_EAFZ"
    source_file = "Turkiye_EAFZ/processed/standard_source_20261106.parquet"
    default_start = "2019-01-01"
    default_validation = "2022-01-01"
    default_test = "2023-01-01"
    default_end = "2024-03-01"
    magnitude_resolution = 0.01
    time_resolution_seconds = 0.01


@Catalog.register("Turkiye_EAFZ_GFT90_20261106-Standard")
class EAFZMc31(EAFZView):
    dataset_name = "Turkiye_EAFZ_GFT90_20261106"
    default_mc = source_floor = 3.1


@Catalog.register("Turkiye_EAFZ_GFT95_20261106-Standard")
class EAFZMc35(EAFZView):
    dataset_name = "Turkiye_EAFZ_GFT95_20261106"
    default_mc = source_floor = 3.5


@Catalog.register("Greece_Santorini_GFT95_20261106-Standard")
class SantoriniMc26(RemediatedCatalog):
    dataset_name = "Greece_Santorini_GFT95_20261106"
    provider = "Greece_SantoriniSwarm"
    source_file = "Greece_SantoriniSwarm/processed/standard_source_20261106.parquet"
    default_start = "2025-02-05"
    default_validation = "2025-02-12"
    default_test = "2025-02-19"
    default_end = "2025-03-01"
    default_mc = source_floor = 2.6
    magnitude_resolution = 0.01
    time_resolution_seconds = 0.000001
