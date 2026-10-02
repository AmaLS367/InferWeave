"""InferWeave: Unified AI inference deployment SDK across cloud GPU providers."""

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Adapters
    "HttpxHealthcheckProbeAdapter": ("inferweave.adapters.healthcheck", "HttpxHealthcheckProbeAdapter"),
    "MockHealthcheckProbeAdapter": ("inferweave.adapters.healthcheck", "MockHealthcheckProbeAdapter"),
    "AsyncioWatchdogAdapter": ("inferweave.adapters.lifecycle", "AsyncioWatchdogAdapter"),
    "InMemoryDeploymentRepository": ("inferweave.adapters.lifecycle", "InMemoryDeploymentRepository"),
    "JsonDeploymentRepository": ("inferweave.adapters.lifecycle", "JsonDeploymentRepository"),
    "MockWatchdogAdapter": ("inferweave.adapters.lifecycle", "MockWatchdogAdapter"),
    "SqliteDeploymentRepository": ("inferweave.adapters.lifecycle", "SqliteDeploymentRepository"),
    # Auth, clients, inference errors
    "CompositeEndpointAuth": ("inferweave.adapters.auth", "CompositeEndpointAuth"),
    "ModalProxyAuth": ("inferweave.adapters.auth", "ModalProxyAuth"),
    "NoEndpointAuth": ("inferweave.adapters.auth", "NoEndpointAuth"),
    "StaticHeaderAuth": ("inferweave.adapters.auth", "StaticHeaderAuth"),
    "FishSpeechClient": ("inferweave.clients", "FishSpeechClient"),
    "ImageGenerationClient": ("inferweave.clients", "ImageGenerationClient"),
    "InferenceClient": ("inferweave.clients", "InferenceClient"),
    "InferenceConfig": ("inferweave.clients", "InferenceConfig"),
    "InferenceTransport": ("inferweave.clients", "InferenceTransport"),
    "ReferenceAudio": ("inferweave.clients", "ReferenceAudio"),
    "AmbiguousDeploymentError": ("inferweave.core.exceptions", "AmbiguousDeploymentError"),
    "DeploymentNotActiveError": ("inferweave.core.exceptions", "DeploymentNotActiveError"),
    "EndpointNotReadyError": ("inferweave.core.exceptions", "EndpointNotReadyError"),
    "InferenceError": ("inferweave.core.exceptions", "InferenceError"),
    "InferenceTimeoutError": ("inferweave.core.exceptions", "InferenceTimeoutError"),
    "InvalidInferenceResponseError": ("inferweave.core.exceptions", "InvalidInferenceResponseError"),
    "ProviderAuthError": ("inferweave.core.exceptions", "ProviderAuthError"),
    "UnsupportedWorkloadError": ("inferweave.core.exceptions", "UnsupportedWorkloadError"),
    "EndpointAuthPort": ("inferweave.ports.auth", "EndpointAuthPort"),
    # Core Exceptions
    "DeploymentError": ("inferweave.core.exceptions", "DeploymentError"),
    "DeploymentNotFoundError": ("inferweave.core.exceptions", "DeploymentNotFoundError"),
    "HealthcheckError": ("inferweave.core.exceptions", "HealthcheckError"),
    "HealthcheckFailedError": ("inferweave.core.exceptions", "HealthcheckFailedError"),
    "HealthcheckTimeoutError": ("inferweave.core.exceptions", "HealthcheckTimeoutError"),
    "InferWeaveError": ("inferweave.core.exceptions", "InferWeaveError"),
    "InsufficientVramError": ("inferweave.core.exceptions", "InsufficientVramError"),
    "ModelNotFoundError": ("inferweave.core.exceptions", "ModelNotFoundError"),
    "NoFeasibleProviderError": ("inferweave.core.exceptions", "NoFeasibleProviderError"),
    "ProviderNotFoundError": ("inferweave.core.exceptions", "ProviderNotFoundError"),
    "ProviderPlatformError": ("inferweave.core.exceptions", "ProviderPlatformError"),
    "UnknownGpuError": ("inferweave.core.exceptions", "UnknownGpuError"),
    # Domain
    "DeploymentRecord": ("inferweave.domain.deployment_record", "DeploymentRecord"),
    "HardwareValidator": ("inferweave.domain.hardware_validator", "HardwareValidator"),
    "HealthEvaluator": ("inferweave.domain.healthcheck", "HealthEvaluator"),
    "ProbeOutcome": ("inferweave.domain.healthcheck", "ProbeOutcome"),
    "ProbeResult": ("inferweave.domain.healthcheck", "ProbeResult"),
    "ReadinessReport": ("inferweave.domain.healthcheck", "ReadinessReport"),
    "ReadinessState": ("inferweave.domain.healthcheck", "ReadinessState"),
    "AutostopAction": ("inferweave.domain.lifecycle", "AutostopAction"),
    "AutostopPolicy": ("inferweave.domain.lifecycle", "AutostopPolicy"),
    "DeploymentLifecycleEvaluator": ("inferweave.domain.lifecycle", "DeploymentLifecycleEvaluator"),
    "LifecycleState": ("inferweave.domain.lifecycle", "LifecycleState"),
    "DeploymentOptions": ("inferweave.domain.options", "DeploymentOptions"),
    "ProviderOptions": ("inferweave.domain.options", "ProviderOptions"),
    "RuntimeOptions": ("inferweave.domain.options", "RuntimeOptions"),
    # Models
    "Deployment": ("inferweave.models.deployment", "Deployment"),
    "DeploymentRequest": ("inferweave.models.deployment", "DeploymentRequest"),
    "DeploymentStatus": ("inferweave.models.deployment", "DeploymentStatus"),
    "DeploymentState": ("inferweave.models.enums", "DeploymentState"),
    "ProviderType": ("inferweave.models.enums", "ProviderType"),
    "WorkloadType": ("inferweave.models.enums", "WorkloadType"),
    "HardwareRequirements": ("inferweave.models.profile", "HardwareRequirements"),
    "HealthcheckConfig": ("inferweave.models.profile", "HealthcheckConfig"),
    "ModelProfile": ("inferweave.models.profile", "ModelProfile"),
    "GpuSpec": ("inferweave.models.routing", "GpuSpec"),
    "InstanceOffer": ("inferweave.models.routing", "InstanceOffer"),
    "RankedOffer": ("inferweave.models.routing", "RankedOffer"),
    "RoutingConstraints": ("inferweave.models.routing", "RoutingConstraints"),
    "RoutingDecision": ("inferweave.models.routing", "RoutingDecision"),
    "VramCheckResult": ("inferweave.models.routing", "VramCheckResult"),
    # Ports
    "DeploymentRepositoryPort": ("inferweave.ports.deployment_repository", "DeploymentRepositoryPort"),
    "HealthcheckProbePort": ("inferweave.ports.healthcheck", "HealthcheckProbePort"),
    "AutostopWatchdogPort": ("inferweave.ports.lifecycle", "AutostopWatchdogPort"),
    "ComputeProviderPort": ("inferweave.ports.provider", "ComputeProviderPort"),
    # Registry & SDK & Services
    "ModelRegistry": ("inferweave.registry.base", "ModelRegistry"),
    "InferWeave": ("inferweave.sdk", "InferWeave"),
    "HealthcheckService": ("inferweave.services.healthcheck_service", "HealthcheckService"),
    "LifecycleService": ("inferweave.services.lifecycle_service", "LifecycleService"),
}

__all__ = [
    "AmbiguousDeploymentError",
    "AsyncioWatchdogAdapter",
    "AutostopAction",
    "AutostopPolicy",
    "AutostopWatchdogPort",
    "CompositeEndpointAuth",
    "ComputeProviderPort",
    "Deployment",
    "DeploymentError",
    "DeploymentLifecycleEvaluator",
    "DeploymentNotActiveError",
    "DeploymentNotFoundError",
    "DeploymentOptions",
    "DeploymentRecord",
    "DeploymentRepositoryPort",
    "DeploymentRequest",
    "DeploymentState",
    "DeploymentStatus",
    "EndpointAuthPort",
    "EndpointNotReadyError",
    "FishSpeechClient",
    "GpuSpec",
    "HardwareRequirements",
    "HardwareValidator",
    "HealthEvaluator",
    "HealthcheckConfig",
    "HealthcheckError",
    "HealthcheckFailedError",
    "HealthcheckProbePort",
    "HealthcheckService",
    "HealthcheckTimeoutError",
    "HttpxHealthcheckProbeAdapter",
    "ImageGenerationClient",
    "InMemoryDeploymentRepository",
    "InferWeave",
    "InferWeaveError",
    "InferenceClient",
    "InferenceConfig",
    "InferenceError",
    "InferenceTimeoutError",
    "InferenceTransport",
    "InstanceOffer",
    "InsufficientVramError",
    "InvalidInferenceResponseError",
    "JsonDeploymentRepository",
    "LifecycleService",
    "LifecycleState",
    "MockHealthcheckProbeAdapter",
    "MockWatchdogAdapter",
    "ModalProxyAuth",
    "ModelNotFoundError",
    "ModelProfile",
    "ModelRegistry",
    "NoEndpointAuth",
    "NoFeasibleProviderError",
    "ProbeOutcome",
    "ProbeResult",
    "ProviderAuthError",
    "ProviderNotFoundError",
    "ProviderOptions",
    "ProviderPlatformError",
    "ProviderType",
    "RankedOffer",
    "ReadinessReport",
    "ReadinessState",
    "ReferenceAudio",
    "RoutingConstraints",
    "RoutingDecision",
    "RuntimeOptions",
    "SqliteDeploymentRepository",
    "StaticHeaderAuth",
    "UnknownGpuError",
    "UnsupportedWorkloadError",
    "VramCheckResult",
    "WorkloadType",
]



def __getattr__(name: str) -> Any:
    """Lazy imports module attributes upon first access to prevent eagerly loading control-plane dependencies."""
    if name in _LAZY_IMPORTS:
        module_path, symbol_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        val = getattr(mod, symbol_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from inferweave.adapters.auth import (
        CompositeEndpointAuth,
        ModalProxyAuth,
        NoEndpointAuth,
        StaticHeaderAuth,
    )
    from inferweave.adapters.healthcheck import (
        HttpxHealthcheckProbeAdapter,
        MockHealthcheckProbeAdapter,
    )
    from inferweave.adapters.lifecycle import (
        AsyncioWatchdogAdapter,
        InMemoryDeploymentRepository,
        JsonDeploymentRepository,
        MockWatchdogAdapter,
        SqliteDeploymentRepository,
    )
    from inferweave.clients import (
        FishSpeechClient,
        ImageGenerationClient,
        InferenceClient,
        InferenceConfig,
        InferenceTransport,
        ReferenceAudio,
    )
    from inferweave.core.exceptions import (
        AmbiguousDeploymentError,
        DeploymentError,
        DeploymentNotActiveError,
        DeploymentNotFoundError,
        EndpointNotReadyError,
        HealthcheckError,
        HealthcheckFailedError,
        HealthcheckTimeoutError,
        InferenceError,
        InferenceTimeoutError,
        InferWeaveError,
        InsufficientVramError,
        InvalidInferenceResponseError,
        ModelNotFoundError,
        NoFeasibleProviderError,
        ProviderAuthError,
        ProviderNotFoundError,
        ProviderPlatformError,
        UnknownGpuError,
        UnsupportedWorkloadError,
    )
    from inferweave.domain.deployment_record import DeploymentRecord
    from inferweave.domain.hardware_validator import HardwareValidator
    from inferweave.domain.healthcheck import (
        HealthEvaluator,
        ProbeOutcome,
        ProbeResult,
        ReadinessReport,
        ReadinessState,
    )
    from inferweave.domain.lifecycle import (
        AutostopAction,
        AutostopPolicy,
        DeploymentLifecycleEvaluator,
        LifecycleState,
    )
    from inferweave.domain.options import (
        DeploymentOptions,
        ProviderOptions,
        RuntimeOptions,
    )
    from inferweave.models.deployment import (
        Deployment,
        DeploymentRequest,
        DeploymentStatus,
    )
    from inferweave.models.enums import DeploymentState, ProviderType, WorkloadType
    from inferweave.models.profile import (
        HardwareRequirements,
        HealthcheckConfig,
        ModelProfile,
    )
    from inferweave.models.routing import (
        GpuSpec,
        InstanceOffer,
        RankedOffer,
        RoutingConstraints,
        RoutingDecision,
        VramCheckResult,
    )
    from inferweave.ports.auth import EndpointAuthPort
    from inferweave.ports.deployment_repository import DeploymentRepositoryPort
    from inferweave.ports.healthcheck import HealthcheckProbePort
    from inferweave.ports.lifecycle import AutostopWatchdogPort
    from inferweave.ports.provider import ComputeProviderPort
    from inferweave.registry.base import ModelRegistry
    from inferweave.sdk import InferWeave
    from inferweave.services.healthcheck_service import HealthcheckService
    from inferweave.services.lifecycle_service import LifecycleService
