from inferweave.ports.provider import (
    ComputeProviderPort,
    ProvisionResult,
    ResourceStatus,
)


class ComputeProvider(ComputeProviderPort):
    """Base class implemented by all infrastructure backends (SkyPilot, Modal, Lightning)."""


__all__ = ["ComputeProvider", "ProvisionResult", "ResourceStatus"]
