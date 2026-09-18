"""Domain enumerations for workloads, providers, and deployment states."""

from enum import Enum


class WorkloadType(str, Enum):
    """Supported AI workload domains."""

    AUDIO = "audio"
    LLM = "llm"
    IMAGE = "image"
    VIDEO = "video"
    CUSTOM = "custom"


class ProviderType(str, Enum):
    """Underlying infrastructure provider backends."""

    SKYPILOT = "skypilot"
    MODAL = "modal"
    LOCAL_DOCKER = "docker"
    AUTO = "auto"


class DeploymentState(str, Enum):
    """Lifecycle states of a deployment."""

    PENDING = "pending"
    PROVISIONING = "provisioning"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"
