"""Regression tests: provider routing stays async end-to-end on the SDK deploy path."""

import asyncio
import concurrent.futures
from unittest.mock import patch

import pytest

from inferweave import (
    HardwareRequirements,
    InferWeave,
    ModelProfile,
    NoFeasibleProviderError,
    WorkloadType,
)
from inferweave.adapters.catalog.static_catalog import StaticCatalogAdapter
from inferweave.adapters.lifecycle.memory_repository import InMemoryDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.routing import GpuSpec, InstanceOffer
from inferweave.providers.router import ProviderRouter
from inferweave.services.lifecycle_service import LifecycleService
from inferweave.services.routing_service import SmartRoutingService

L4 = GpuSpec(name="L4", vram_gb=24.0)
A100 = GpuSpec(name="A100-80GB", vram_gb=80.0)


def _offers() -> list[InstanceOffer]:
    return [
        InstanceOffer(provider="modal", instance_type="modal.l4", gpu_spec=L4, price_per_hour=0.80),
        InstanceOffer(provider="runpod", instance_type="runpod.l4", gpu_spec=L4, price_per_hour=0.40),
        InstanceOffer(provider="aws", instance_type="aws.a100", gpu_spec=A100, price_per_hour=4.10),
    ]


def _profile(recommended: list[str] | None = None) -> ModelProfile:
    return ModelProfile(
        id="async-routing-test",
        name="Async Routing Test",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(min_vram_gb=16.0, recommended_gpus=recommended or []),
    )


def _router(offers: list[InstanceOffer] | None = None) -> ProviderRouter:
    return ProviderRouter(
        routing_service=SmartRoutingService(
            catalog=StaticCatalogAdapter(custom_offers=offers or _offers())
        )
    )


def _weave() -> InferWeave:
    # In-memory lifecycle storage keeps the async path free of unrelated thread-pool usage
    # (e.g. the SQLite repository), so the thread guards below only trip on routing.
    lifecycle = LifecycleService(
        watchdog_port=MockWatchdogAdapter(), repository=InMemoryDeploymentRepository()
    )
    return InferWeave(router=_router(), lifecycle_service=lifecycle)


def _forbid_sync_bridge():
    """Patches every way the old sync bridge could run: wrappers, nested loops, and threads."""

    def _boom(*_a, **_k):
        raise AssertionError("synchronous routing bridge invoked from the async path")

    return (
        patch.object(SmartRoutingService, "resolve", _boom),
        patch.object(ProviderRouter, "resolve", _boom),
        patch.object(asyncio, "run", _boom),
        patch.object(concurrent.futures, "ThreadPoolExecutor", _boom),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["auto", "runpod"])
async def test_sdk_deploy_routes_via_aresolve_without_sync_bridge(provider):
    weave = _weave()
    p1, p2, p3, p4 = _forbid_sync_bridge()
    with (
        p1,
        p2,
        p3,
        p4,
        patch.object(
            SmartRoutingService, "aresolve", autospec=True, side_effect=SmartRoutingService.aresolve
        ) as spy,
    ):
        weave.register_model(_profile())
        deployment = await weave.deploy(
            model="async-routing-test", provider=provider, dry_run=True
        )

    spy.assert_awaited_once()
    assert weave.router.last_decision is not None
    assert deployment.provider == ("runpod" if provider == "auto" else provider)


@pytest.mark.asyncio
async def test_sdk_deploy_with_explicit_gpu_skips_routing_entirely():
    weave = _weave()
    weave.register_model(_profile())
    with patch.object(SmartRoutingService, "aresolve", side_effect=AssertionError("no routing")):
        deployment = await weave.deploy(
            model="async-routing-test", provider="runpod", gpu_type="L4", dry_run=True
        )
    assert deployment.provider == "runpod"


@pytest.mark.asyncio
async def test_sync_router_resolve_refuses_to_run_inside_event_loop():
    router = _router()
    with pytest.raises(RuntimeError, match=r"ProviderRouter\.aresolve"):
        router.resolve("auto", _profile())


@pytest.mark.asyncio
async def test_sync_routing_service_resolve_refuses_to_run_inside_event_loop():
    service = SmartRoutingService(catalog=StaticCatalogAdapter(custom_offers=_offers()))
    request = DeploymentRequest(model="async-routing-test", provider="auto", dry_run=True)
    with pytest.raises(RuntimeError, match=r"SmartRoutingService\.aresolve"):
        service.resolve(_profile(), request)


def test_sync_resolve_outside_event_loop_matches_async_selection():
    """The sync API remains usable for non-async callers and selects identically."""
    sync_router, async_router = _router(), _router()
    sync_req = DeploymentRequest(
        model="async-routing-test", provider="auto", strategy="cheapest", dry_run=True
    )
    async_req = DeploymentRequest(
        model="async-routing-test", provider="auto", strategy="cheapest", dry_run=True
    )

    sync_provider = sync_router.resolve("auto", _profile(), request=sync_req)
    async_provider = asyncio.run(async_router.aresolve("auto", _profile(), request=async_req))

    assert sync_provider.name == async_provider.name == "runpod"
    assert sync_req.gpu_type == async_req.gpu_type == "L4"
    assert sync_router.last_decision is not None
    assert sync_router.last_decision.chosen_offer == async_router.last_decision.chosen_offer


@pytest.mark.asyncio
async def test_aresolve_auto_selects_by_strategy_and_fills_gpu():
    router = _router()
    request = DeploymentRequest(
        model="async-routing-test", provider="auto", strategy="cheapest", dry_run=True
    )

    provider = await router.aresolve("auto", _profile(), "cheapest", request=request)

    assert provider.name == "runpod"
    assert request.gpu_type == "L4"
    assert router.last_decision is not None
    assert router.last_decision.strategy_used == "cheapest"


@pytest.mark.asyncio
async def test_aresolve_provider_scoped_routing_picks_gpu_for_that_provider():
    router = _router()
    request = DeploymentRequest(model="async-routing-test", provider="modal", dry_run=True)

    provider = await router.aresolve("modal", _profile(), request=request)

    assert provider.name == "modal"
    assert request.gpu_type == "L4"
    assert router.last_decision is not None
    assert router.last_decision.chosen_provider == "modal"


@pytest.mark.asyncio
async def test_aresolve_rejected_provider_without_catalog_uses_recommended_gpu():
    """A provider with no catalog offers falls back to the profile's recommended GPUs."""
    router = _router()
    request = DeploymentRequest(model="async-routing-test", provider="fluidstack", dry_run=True)

    provider = await router.aresolve("fluidstack", _profile(["A100-80GB"]), request=request)

    assert provider.name == "fluidstack"
    assert request.gpu_type == "A100-80GB"


@pytest.mark.asyncio
async def test_aresolve_infeasible_hardware_still_raises():
    router = _router()
    big = _profile()
    big.hardware.min_vram_gb = 500.0
    request = DeploymentRequest(model="async-routing-test", provider="auto", dry_run=True)

    with pytest.raises(NoFeasibleProviderError):
        await router.aresolve("auto", big, request=request)
