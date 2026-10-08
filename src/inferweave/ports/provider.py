"""Port contract for cloud and container compute infrastructure backends.

Providers are stateless remote adapters: every operation receives the deployment record (its
nonsecret identity) and the :class:`ProviderAccount` that owns it, and must use exactly those
credentials. Persistence, account scheduling, failover and reconciliation are owned by the
application services, never by providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from inferweave.accounts.models import ProviderAccount
from inferweave.domain.deployment_record import DeploymentRecord, ResourceRef
from inferweave.domain.lifecycle import AutostopAction
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.runtimes.base import RuntimeSpec


@dataclass(frozen=True)
class ProvisionResult:
    """What a successful create call established (infrastructure only, not readiness)."""

    state: DeploymentState
    endpoint_url: str | None
    resource_id: str | None = None


@dataclass(frozen=True)
class ResourceStatus:
    """Infrastructure-level state of an existing remote resource."""

    state: DeploymentState
    endpoint_url: str | None = None


class ComputeProviderPort(ABC):
    """Abstract port interface for infrastructure backends (SkyPilot clouds, Modal, Lightning)."""

    cleanup_failed_deployment = False
    """Whether a deployment that never became ready should be destroyed automatically."""

    async def prepare(self, record: DeploymentRecord, account: ProviderAccount) -> None:
        """Read-only validation before the final ownership record is persisted and create starts."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for this provider instance (e.g. 'runpod', 'modal', 'aws')."""

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Category of this compute backend."""

    @abstractmethod
    def new_deployment_id(self, profile: ModelProfile) -> str:
        """A fresh deployment id; it is also the remote resource name (unique, deterministic)."""

    @abstractmethod
    def resource_ref(
        self, deployment_id: str, request: DeploymentRequest, account: ProviderAccount
    ) -> ResourceRef:
        """Nonsecret identity of the remote resource, persisted before it is created."""

    def preflight(
        self,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> None:
        """Local validation before any remote call (SDK present, credentials complete...)."""

    def dry_run_endpoint(self, deployment_id: str, runtime: RuntimeSpec) -> str | None:
        """Placeholder endpoint reported by dry-run deployments."""
        return None

    @abstractmethod
    async def provision(
        self,
        record: DeploymentRecord,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> ProvisionResult:
        """Creates the remote resource named ``record.resource.name`` under ``account``.

        Raises :class:`~inferweave.core.exceptions.ProviderOperationError` with a precise
        ``kind`` (and ``resource_may_exist``) whenever the failure can be classified.
        """

    @abstractmethod
    async def status(self, record: DeploymentRecord, account: ProviderAccount) -> ResourceStatus:
        """Infrastructure state of the deployment (``STOPPED`` when the resource is gone)."""

    @abstractmethod
    async def resource_exists(self, record: DeploymentRecord, account: ProviderAccount) -> bool:
        """Whether the remote resource exists; used to reconcile uncertain operations."""

    @abstractmethod
    async def stop(
        self,
        record: DeploymentRecord,
        account: ProviderAccount,
        action: AutostopAction = AutostopAction.STOP,
    ) -> None:
        """Stops/destroys the remote resource; a resource that is already gone is a success."""
