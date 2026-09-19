"""Composite catalog adapter delegating to primary sources with fallback to static catalog."""

import logging

from inferweave.models.routing import InstanceOffer
from inferweave.ports.catalog import ProviderCatalogPort

logger = logging.getLogger(__name__)


class CompositeCatalogAdapter(ProviderCatalogPort):
    """Queries a primary catalog source, falling back to a static catalog if the primary fails."""

    def __init__(
        self,
        primary_catalog: ProviderCatalogPort,
        fallback_catalog: ProviderCatalogPort,
    ) -> None:
        self._primary = primary_catalog
        self._fallback = fallback_catalog

    async def get_offers(self, provider_name: str | None = None) -> list[InstanceOffer]:
        try:
            offers = await self._primary.get_offers(provider_name=provider_name)
            if offers:
                return offers
        except Exception as err:  # noqa: BLE001
            logger.debug(
                "Primary catalog failed (%s), falling back to secondary catalog", err
            )

        return await self._fallback.get_offers(provider_name=provider_name)
