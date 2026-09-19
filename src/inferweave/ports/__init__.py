"""Clean Architecture Port definitions."""

from inferweave.ports.catalog import AvailabilityProbePort, ProviderCatalogPort
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.ports.gpu_catalog import GpuCatalogPort
from inferweave.ports.healthcheck import HealthcheckProbePort
from inferweave.ports.lifecycle import AutostopWatchdogPort
from inferweave.ports.provider import ComputeProviderPort

__all__ = [
    "AutostopWatchdogPort",
    "AvailabilityProbePort",
    "ComputeProviderPort",
    "DeploymentRepositoryPort",
    "GpuCatalogPort",
    "HealthcheckProbePort",
    "ProviderCatalogPort",
]
