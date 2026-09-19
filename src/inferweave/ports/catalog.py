"""Port interfaces for compute provider catalogs and availability probes."""

from abc import ABC, abstractmethod

from inferweave.models.routing import InstanceOffer


class ProviderCatalogPort(ABC):
    """Abstract port for querying compute provider hardware instances and pricing."""

    @abstractmethod
    async def get_offers(self, provider_name: str | None = None) -> list[InstanceOffer]:
        """Returns a list of available instance configurations and prices.

        Args:
            provider_name: Optional provider name filter (e.g. 'runpod', 'modal').
        """


class AvailabilityProbePort(ABC):
    """Abstract port for checking live availability of GPU instances."""

    @abstractmethod
    async def check_availability(
        self,
        provider: str,
        gpu_type: str,
        region: str | None = None,
    ) -> bool:
        """Returns True if the requested GPU is currently in-stock and launchable."""
