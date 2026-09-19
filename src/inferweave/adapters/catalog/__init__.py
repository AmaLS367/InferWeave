"""Catalog adapters package."""

from inferweave.adapters.catalog.composite_catalog import CompositeCatalogAdapter
from inferweave.adapters.catalog.static_catalog import StaticCatalogAdapter
from inferweave.adapters.catalog.static_gpu_catalog import StaticGpuCatalogAdapter

__all__ = [
    "CompositeCatalogAdapter",
    "StaticCatalogAdapter",
    "StaticGpuCatalogAdapter",
]
