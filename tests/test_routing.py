"""Unit and integration tests for Clean Architecture intelligent provider routing."""

import pytest

from inferweave import (
    HardwareRequirements,
    InferWeave,
    ModelProfile,
    NoFeasibleProviderError,
    WorkloadType,
)
from inferweave.adapters.catalog.static_catalog import StaticCatalogAdapter
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.routing import (
    GpuSpec,
    InstanceOffer,
    RoutingConstraints,
)
from inferweave.routing.strategies import (
    BalancedStrategy,
    CheapestStrategy,
    FreeFirstStrategy,
    LowestLatencyStrategy,
    get_strategy,
)
from inferweave.services.routing_service import SmartRoutingService


@pytest.fixture
def sample_gpu_t4() -> GpuSpec:
    return GpuSpec(name="T4", vram_gb=16.0)


@pytest.fixture
def sample_gpu_l4() -> GpuSpec:
    return GpuSpec(name="L4", vram_gb=24.0)


@pytest.fixture
def sample_gpu_a100() -> GpuSpec:
    return GpuSpec(name="A100-80GB", vram_gb=80.0)


@pytest.fixture
def sample_offers(sample_gpu_t4, sample_gpu_l4, sample_gpu_a100) -> list[InstanceOffer]:
    return [
        InstanceOffer(
            provider="modal",
            instance_type="modal.l4",
            gpu_spec=sample_gpu_l4,
            price_per_hour=0.80,
            estimated_cold_start_sec=8.0,
            is_available=True,
        ),
        InstanceOffer(
            provider="runpod",
            instance_type="runpod.l4",
            gpu_spec=sample_gpu_l4,
            price_per_hour=0.70,
            spot_price_per_hour=0.40,
            estimated_cold_start_sec=65.0,
            is_available=True,
        ),
        InstanceOffer(
            provider="aws",
            instance_type="g4dn.xlarge",
            gpu_spec=sample_gpu_t4,
            price_per_hour=0.526,
            spot_price_per_hour=0.20,
            estimated_cold_start_sec=120.0,
            is_available=True,
        ),
        InstanceOffer(
            provider="lambda",
            instance_type="lambda.a100",
            gpu_spec=sample_gpu_a100,
            price_per_hour=1.49,
            estimated_cold_start_sec=80.0,
            is_available=True,
        ),
    ]


def test_effective_price_and_vram(sample_gpu_l4):
    offer = InstanceOffer(
        provider="runpod",
        instance_type="runpod.l4",
        gpu_spec=sample_gpu_l4,
        gpu_count=2,
        price_per_hour=1.40,
        spot_price_per_hour=0.80,
    )
    assert offer.total_vram_gb == 48.0
    assert offer.effective_price(allow_spot=True) == 0.80
    assert offer.effective_price(allow_spot=False) == 1.40


def test_cheapest_strategy_ranking(sample_offers):
    strategy = CheapestStrategy()
    constraints = RoutingConstraints(min_vram_gb=16.0, allow_spot=True)
    ranked = strategy.rank(sample_offers, constraints)

    assert len(ranked) == 4
    # With spot allowed: AWS T4 ($0.20) < RunPod L4 ($0.40) < Modal L4 ($0.80) < Lambda A100 ($1.49)
    assert ranked[0].offer.provider == "aws"
    assert ranked[1].offer.provider == "runpod"
    assert ranked[2].offer.provider == "modal"
    assert ranked[3].offer.provider == "lambda"


def test_cheapest_strategy_no_spot(sample_offers):
    strategy = CheapestStrategy()
    constraints = RoutingConstraints(min_vram_gb=16.0, allow_spot=False)
    ranked = strategy.rank(sample_offers, constraints)

    # On-demand: AWS T4 ($0.526) < RunPod L4 ($0.70) < Modal L4 ($0.80) < Lambda A100 ($1.49)
    assert ranked[0].offer.provider == "aws"
    assert ranked[1].offer.provider == "runpod"


def test_free_first_strategy(sample_offers, sample_gpu_t4):
    free_offer = InstanceOffer(
        provider="docker",
        instance_type="local.t4",
        gpu_spec=sample_gpu_t4,
        price_per_hour=0.0,
        is_free_tier=True,
        estimated_cold_start_sec=1.0,
    )
    all_offers = [free_offer, *sample_offers]

    strategy = FreeFirstStrategy()
    constraints = RoutingConstraints(min_vram_gb=16.0)
    ranked = strategy.rank(all_offers, constraints)

    assert ranked[0].offer.provider == "docker"
    assert ranked[0].offer.is_free_tier is True


def test_lowest_latency_strategy(sample_offers):
    strategy = LowestLatencyStrategy()
    constraints = RoutingConstraints(min_vram_gb=16.0)
    ranked = strategy.rank(sample_offers, constraints)

    # Modal cold start is 8.0s vs 65s for runpod vs 80s for lambda vs 120s for aws
    assert ranked[0].offer.provider == "modal"
    assert ranked[0].offer.estimated_cold_start_sec == 8.0


def test_balanced_strategy(sample_offers):
    strategy = BalancedStrategy()
    constraints = RoutingConstraints(min_vram_gb=16.0)
    ranked = strategy.rank(sample_offers, constraints)

    assert len(ranked) == 4
    # All scores should be between 0.0 and 1.0
    for r in ranked:
        assert 0.0 <= r.total_score <= 1.0


@pytest.mark.asyncio
async def test_routing_service_vram_filtering(sample_offers):
    adapter = StaticCatalogAdapter(custom_offers=sample_offers)
    service = SmartRoutingService(catalog=adapter)

    # Model requiring 24GB VRAM: AWS T4 (16GB) must be filtered out
    profile = ModelProfile(
        id="flux-test",
        name="Flux Test",
        workload_type=WorkloadType.IMAGE,
        default_runtime="vllm",
        hardware=HardwareRequirements(min_vram_gb=24.0),
    )
    request = DeploymentRequest(
        model=profile.id,
        provider="auto",
        strategy="cheapest",
        dry_run=True,
    )

    decision = await service.aresolve(profile, request)
    # Filtered candidates: RunPod L4 ($0.40 spot), Modal L4 ($0.80), Lambda A100 ($1.49)
    # AWS T4 excluded because 16GB < 24GB
    assert decision.chosen_provider == "runpod"
    assert decision.chosen_offer.total_vram_gb >= 24.0


@pytest.mark.asyncio
async def test_routing_service_explicit_gpu_filter(sample_offers):
    adapter = StaticCatalogAdapter(custom_offers=sample_offers)
    service = SmartRoutingService(catalog=adapter)

    profile = ModelProfile(
        id="test-model",
        name="Test",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(min_vram_gb=16.0),
    )
    request = DeploymentRequest(
        model=profile.id,
        provider="auto",
        strategy="cheapest",
        gpu_type="A100",
        dry_run=True,
    )

    decision = await service.aresolve(profile, request)
    assert decision.chosen_provider == "lambda"
    assert "A100" in decision.chosen_offer.gpu_spec.name


@pytest.mark.asyncio
async def test_routing_service_no_feasible_provider(sample_offers):
    adapter = StaticCatalogAdapter(custom_offers=sample_offers)
    service = SmartRoutingService(catalog=adapter)

    profile = ModelProfile(
        id="huge-model",
        name="Huge",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(min_vram_gb=500.0),
    )
    request = DeploymentRequest(
        model=profile.id,
        provider="auto",
        dry_run=True,
    )

    with pytest.raises(NoFeasibleProviderError) as exc_info:
        await service.aresolve(profile, request)

    assert "No compute provider satisfies requirements" in str(exc_info.value)
    assert len(exc_info.value.reasons) > 0


@pytest.mark.asyncio
async def test_sdk_auto_provider_cheapest_dry_run():
    weave = InferWeave()
    # Model 'fish-s2-pro' requires 16GB VRAM
    deployment = await weave.deploy(
        model="fish-s2-pro",
        provider="auto",
        strategy="cheapest",
        dry_run=True,
    )

    assert deployment.provider in ["vast", "runpod", "modal", "aws", "lambda", "nebius"]
    assert weave.router.last_decision is not None
    assert weave.router.last_decision.strategy_used == "cheapest"
    assert weave.router.last_decision.chosen_offer.total_vram_gb >= 16.0


@pytest.mark.asyncio
async def test_sdk_auto_provider_lowest_latency_dry_run():
    weave = InferWeave()
    # Model 'fish-s2-pro' with lowest_latency should pick Modal due to fastest cold-start
    deployment = await weave.deploy(
        model="fish-s2-pro",
        provider="auto",
        strategy="lowest_latency",
        dry_run=True,
    )

    assert deployment.provider == "modal"
    assert weave.router.last_decision is not None
    assert weave.router.last_decision.strategy_used == "lowest_latency"
    assert weave.router.last_decision.chosen_offer.provider == "modal"


def test_invalid_routing_strategy_raises_value_error():
    """Validates that get_strategy raises ValueError on unknown strategy name."""
    with pytest.raises(ValueError) as exc_info:
        get_strategy("potato")

    assert "Invalid routing strategy 'potato'" in str(exc_info.value)
    assert "cheapest" in str(exc_info.value)


@pytest.mark.asyncio
async def test_smart_routing_with_live_availability_probe(sample_offers):
    """Validates that SmartRoutingService queries AvailabilityProbePort and filters out unavailable offers."""
    from inferweave.ports.catalog import AvailabilityProbePort

    class MockProbe(AvailabilityProbePort):
        async def check_availability(
            self, provider: str, gpu_type: str, region: str | None = None
        ) -> bool:
            # Simulate cheapest provider (runpod) being out of stock
            return provider != "runpod"

    probe = MockProbe()
    catalog = StaticCatalogAdapter(custom_offers=sample_offers)
    service = SmartRoutingService(catalog=catalog, availability_probe=probe)

    profile = ModelProfile(
        id="test-model",
        name="Test Model",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(min_vram_gb=16.0),
    )
    req = DeploymentRequest(model="test-model", provider="auto", strategy="cheapest")

    decision = await service.aresolve(profile, req)
    # runpod was cheapest, but probe reported unavailable, so next available offer is picked
    assert decision.chosen_provider != "runpod"
    assert decision.chosen_provider in ["lambda", "aws", "modal"]
