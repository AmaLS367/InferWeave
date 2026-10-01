"""Runtime specification and template base contracts."""

import shlex
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, model_validator

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import ModelProfile


class RuntimeSpec(BaseModel):
    """Concrete container configuration produced by a runtime template."""

    name: str = Field(..., description="Runtime template identifier, e.g. 'vllm'")
    docker_image: str = Field(..., description="Container image reference: repository with a tag or an immutable @sha256 digest")
    setup_commands: list[str] = Field(
        default_factory=list, description="Commands executed before starting the server"
    )
    run_command: str = Field(..., description="Primary server launch command")
    run_args: list[str] = Field(
        default_factory=list,
        description="Structured argv tokens for server launch without shell interpretation",
    )
    port: int = Field(default=8000, description="Exposed port for inference requests")
    env_vars: dict[str, str] = Field(
        default_factory=dict, description="Environment variables injected into runtime"
    )
    healthcheck_path: str = Field(
        default="/health", description="HTTP endpoint for health monitoring"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _sync_command_and_args(cls, data: Any) -> Any:
        if isinstance(data, dict):
            args = data.get("run_args")
            cmd = data.get("run_command")
            if args and not cmd:
                data["run_command"] = shlex.join([str(a) for a in args])
            elif cmd and not args:
                try:
                    data["run_args"] = shlex.split(str(cmd))
                except ValueError:
                    data["run_args"] = [str(cmd)]
        return data


class RuntimeTemplate(ABC):
    """Abstract recipe for building runtime specifications for a family of models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of this runtime template."""

    @abstractmethod
    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        """Produces a concrete RuntimeSpec for the specified model profile and deployment request."""
