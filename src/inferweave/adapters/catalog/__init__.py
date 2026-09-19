"""Catalog adapters package."""

from inferweave.adapters.catalog.composite_catalog import CompositeCatalogAdapter
from inferweave.adapters.catalog.static_catalog import StaticCatalogAdapter

__all__ = ["CompositeCatalogAdapter", "StaticCatalogAdapter"]
