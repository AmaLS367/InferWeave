"""Domain models and value objects for intelligent provider routing."""

from typing import Any

from pydantic import BaseModel, Field


class GpuSpec(BaseModel):
    """Hardware specifications for a specific GPU model."""

    name: str = Field(..., description="GPU model name, e.g. 'A10G', 'L4', 'A100', 'H100'")
    vram_gb: float = Field(..., description="GPU VRAM in gigabytes")
    architecture: str | None = Field(default=None, description="GPU microarchitecture, e.g. 'Ada Lovelace', 'Ampere', 'Hopper'")
    compute_capability: float | None = Field(default=None, description="CUDA compute capability score, e.g. 8.0, 8.9, 9.0")


class InstanceOffer(BaseModel):
    """Specific GPU cloud instance offer from a compute provider."""

    provider: str = Field(..., description="Compute provider name (e.g. 'modal', 'runpod', 'aws')")
    instance_type: str = Field(..., description="Instance identifier, e.g. 'modal.l4', 'g5.xlarge'")
    gpu_spec: GpuSpec = Field(..., description="GPU hardware specification")
    gpu_count: int = Field(default=1, ge=1, description="Number of GPUs attached to this instance")
    price_per_hour: float = Field(..., ge=0.0, description="On-demand price in USD per hour")
    spot_price_per_hour: float | None = Field(default=None, ge=0.0, description="Spot/preemptible price in USD per hour")
    is_free_tier: bool = Field(default=False, description="Whether this offer is covered by free tier or zero cost")
    is_available: bool = Field(default=True, description="Whether the instance capacity is currently in stock/available")
    estimated_cold_start_sec: float = Field(default=60.0, ge=0.0, description="Typical cold-start latency in seconds")
    billing_granularity_sec: int = Field(default=60, ge=1, description="Billing increment in seconds (e.g. 1s or 60s)")
    region: str | None = Field(default=None, description="Geographic datacenter region, e.g. 'us-east-1'")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific metadata")

    @property
    def total_vram_gb(self) -> float:
        """Total GPU VRAM available on this instance."""
        return self.gpu_spec.vram_gb * self.gpu_count

    def effective_price(self, allow_spot: bool = True) -> float:
        """Returns the effective hourly price considering spot availability."""
        if allow_spot and self.spot_price_per_hour is not None:
            return self.spot_price_per_hour
        return self.price_per_hour


class RoutingConstraints(BaseModel):
    """Constraints and preferences evaluated during provider routing."""

    min_vram_gb: float = Field(..., description="Minimum total GPU VRAM required in gigabytes")
    gpu_count: int = Field(default=1, ge=1, description="Required number of GPUs")
    gpu_type: str | None = Field(default=None, description="Strict GPU model preference if specified by user")
    recommended_gpus: list[str] = Field(default_factory=list, description="Recommended GPU models in order of priority")
    allow_spot: bool = Field(default=True, description="Allow spot/preemptible instances for lower cost")
    max_price_per_hour: float | None = Field(default=None, description="Maximum acceptable hourly price in USD")
    preferred_regions: list[str] = Field(default_factory=list, description="List of preferred datacenter regions")


class RankedOffer(BaseModel):
    """Evaluated instance offer with ranking scores and decision explanation."""

    offer: InstanceOffer = Field(..., description="Underlying provider instance offer")
    total_score: float = Field(..., description="Normalized overall ranking score (higher is better)")
    cost_score: float = Field(default=0.0, description="Cost efficiency score [0.0 - 1.0]")
    latency_score: float = Field(default=0.0, description="Latency/speed score [0.0 - 1.0]")
    availability_score: float = Field(default=0.0, description="Availability confidence score [0.0 - 1.0]")
    reasoning: str = Field(default="", description="Human-readable rationale for this score")


class RoutingDecision(BaseModel):
    """Final output of the provider routing use case."""

    chosen_provider: str = Field(..., description="Name of the selected compute provider")
    chosen_offer: InstanceOffer = Field(..., description="Selected instance offer configuration")
    rankings: list[RankedOffer] = Field(default_factory=list, description="All evaluated offers with their rankings")
    strategy_used: str = Field(default="cheapest", description="Routing strategy name used for selection")
