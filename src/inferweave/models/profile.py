"""Model hardware requirements, healthcheck parameters, and profile definitions."""

from typing import Any

from pydantic import BaseModel, Field

from inferweave.models.enums import WorkloadType


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

    enabled: bool = Field(
        default=True, description="Whether readiness healthcheck polling is enabled"
    )
    path: str = Field(
        default="/health", description="HTTP endpoint path for readiness probe"
    )
    port: int = Field(default=8000, description="Exposed service port")
    initial_delay_seconds: float = Field(
        default=10.0, ge=0.0, description="Delay before first probe check in seconds"
    )
    timeout_seconds: float = Field(
        default=300.0,
        ge=0.01,
        description="Total timeout waiting for model readiness in seconds",
    )
    probe_interval_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Interval between subsequent health checks in seconds",
    )
    request_timeout_seconds: float = Field(
        default=5.0,
        ge=0.5,
        description="Timeout for an individual HTTP probe request in seconds",
    )
    expected_status_codes: list[int] = Field(
        default_factory=lambda: [200],
        description="HTTP status codes considered healthy",
    )
    consecutive_successes: int = Field(
        default=1,
        ge=1,
        description="Consecutive successful probes required for readiness",
    )
    method: str = Field(
        default="GET",
        description="HTTP method for probe, e.g. 'GET' or 'HEAD'",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers sent with probe requests",
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
