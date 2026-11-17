"""Catalog module registry.

Importing this package triggers side-effect registration of catalog classes
decorated with ``@Catalog.register(...)``.
"""

from . import (
    azdx,
    basel,
    ccl,
    china_array,
    chuandian,
    cooper_basin,
    egs_collab,
    forge,
    geysers,
    hauksson,
    pnr,
    qtm,
    scedc,
    ssfs,
    st1,
    synthetic_etas,
    virtual_induced,
    white,
)

__all__ = [
    "azdx",
    "basel",
    "ccl",
    "china_array",
    "chuandian",
    "cooper_basin",
    "egs_collab",
    "forge",
    "geysers",
    "hauksson",
    "pnr",
    "qtm",
    "scedc",
    "ssfs",
    "st1",
    "synthetic_etas",
    "virtual_induced",
    "white",
]

# Public earthquake catalogs prepared from the audited 20261105 snapshots.
from . import external_catalog
__all__.append("external_catalog")

from . import regional_catalogs_20261105
__all__.append("regional_catalogs_20261105")

from . import more_catalogs_20261105
__all__.append("more_catalogs_20261105")

from . import research_catalogs_20261106
__all__.append("research_catalogs_20261106")

from . import earthquakenpp_20261106
__all__.append("earthquakenpp_20261106")

from . import source_expansion_catalogs_20261106
__all__.append("source_expansion_catalogs_20261106")

from . import doi_catalogs_20261106
__all__.append("doi_catalogs_20261106")

from . import remediated_catalogs_20261106
__all__.append("remediated_catalogs_20261106")
