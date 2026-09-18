"""Model hardware requirements, healthcheck parameters, and profile definitions."""

from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import WorkloadType


class HardwareRequirements(BaseModel):
    """Hardware requirements and hints for GPU scheduling."""

    min_vram_gb: float = Field(..., description="Minimum GPU VRAM in gigabytes")
    recommended_gpus: list[str] = Field(
        default_factory=lambda: ["A10G", "L4", "A100", "H100"],
        description="List of recommended GPU types in order of preference",
    )
    gpu_count: int = Field(default=1, ge=1, description="Number of GPUs required")
    min_cpu_cores: int = Field(default=4, ge=1, description="Minimum CPU cores")
    min_ram_gb: float = Field(
        default=16.0, ge=1.0, description="Minimum system RAM in gigabytes"
    )
    min_cuda_version: str | None = Field(
        default="12.1", description="Minimum CUDA version"
    )


class HealthcheckConfig(BaseModel):
    """Parameters for monitoring endpoint readiness."""

    path: str = Field(
        default="/health", description="HTTP endpoint path for readiness probe"
    )
    port: int = Field(default=8000, description="Exposed service port")
    initial_delay_seconds: int = Field(
        default=10, description="Delay before first probe check"
    )
    timeout_seconds: int = Field(
        default=300, description="Total timeout waiting for model readiness"
    )
    probe_interval_seconds: int = Field(
        default=3, description="Interval between subsequent health checks"
    )


class ModelProfile(BaseModel):
    """Metadata and resource profile for a deployable AI model."""

    id: str = Field(..., description="Unique model identifier, e.g. 'fish-s2-pro'")
    name: str = Field(..., description="Human-readable model name")
    workload_type: WorkloadType = Field(
        ..., description="Category of the model workload"
    )
    default_runtime: str = Field(
        ..., description="Identifier of the default runtime template, e.g. 'vllm'"
    )
    hardware: HardwareRequirements = Field(..., description="Hardware prerequisites")
    healthcheck: HealthcheckConfig = Field(default_factory=HealthcheckConfig)
    default_env: dict[str, str] = Field(
        default_factory=dict, description="Default environment variables"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional arbitrary metadata"
    )
