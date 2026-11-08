"""Versioned CWA, Wu et al. and East Anatolian catalog adapters."""
from src.data import Catalog
from .research_catalogs_20261106 import Catalog20261106


@Catalog.register("Taiwan_CWA-Standard")
class TaiwanCWA(Catalog20261106):
    dataset_name = provider = "Taiwan_CWA"
    default_start = "1991-01-01"
    default_end = "2026-01-01"
    default_mc = source_floor = 3.0
    magnitude_resolution = 0.1
    time_resolution_seconds = 0.01
    quality_protocol = "cwa-historical-manual-f-20261106-v1"


@Catalog.register("Taiwan_Wu2008-Standard")
class TaiwanWu2008(Catalog20261106):
    dataset_name = provider = "Taiwan_Wu2008"
    default_start = "1991-01-01"
    default_validation = "2002-01-01"
    default_test = "2004-01-01"
    default_end = "2006-01-01"
    default_mc = source_floor = 2.5
    magnitude_resolution = 0.01
    time_resolution_seconds = 0.01
    quality_protocol = "wu2008-relative-to-cwbsn-research-20261106-v1"


@Catalog.register("Turkiye_EAFZ-Standard")
class TurkiyeEAFZ(Catalog20261106):
    dataset_name = provider = "Turkiye_EAFZ"
    default_start = "2019-01-01"
    default_validation = "2022-01-01"
    default_test = "2023-01-01"
    default_end = "2024-03-01"
    default_mc = source_floor = 2.5
    magnitude_resolution = 0.01
    time_resolution_seconds = 0.01
    quality_protocol = "eafz-casp-nlloc-research-20261106-v1"
