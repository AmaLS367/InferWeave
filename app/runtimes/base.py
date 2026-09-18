"""Runtime specification and template base contracts."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.models.deployment import DeploymentRequest
from app.models.profile import ModelProfile


class RuntimeSpec(BaseModel):
    """Concrete container configuration produced by a runtime template."""
    name: str = Field(..., description="Runtime template identifier, e.g. 'vllm'")
    docker_image: str = Field(..., description="Container image repository and tag")
    setup_commands: list[str] = Field(default_factory=list, description="Commands executed before starting the server")
    run_command: str = Field(..., description="Primary server launch command")
    port: int = Field(default=8000, description="Exposed port for inference requests")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables injected into runtime")
    healthcheck_path: str = Field(default="/health", description="HTTP endpoint for health monitoring")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeTemplate(ABC):
    """Abstract recipe for building runtime specifications for a family of models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of this runtime template."""

    @abstractmethod
    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        """Produces a concrete RuntimeSpec for the specified model profile and deployment request."""
