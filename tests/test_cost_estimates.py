"""Catalog quote provenance and unavailable billing semantics, with fake providers only."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fakes import FakeProvider, fail, make_weave

from inferweave import ComputePricing, CostEstimate
from inferweave.accounts import AccountsConfig
from inferweave.adapters.catalog.static_catalog import StaticCatalogAdapter
from inferweave.adapters.lifecycle import (
    JsonDeploymentRepository,
    SqliteDeploymentRepository,
)
from inferweave.core.exceptions import ProviderOperationError
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.lifecycle import AutostopAction
from inferweave.models.enums import ProviderType
from inferweave.models.routing import GpuSpec, InstanceOffer
from inferweave.models.usage import estimate_cost
from inferweave.services.routing_service import SmartRoutingService


def offer(provider="modal", gpu="L4", count=1, **kwargs):
    return InstanceOffer(
        provider=provider, instance_type=f"{provider}.{gpu}",
        gpu_spec=GpuSpec(name=gpu, vram_gb=24), gpu_count=count,
        price_per_hour=3.6, billing_granularity_sec=60, **kwargs,
    )


def setup_weave(tmp_path, offers, providers):
    weave = make_weave(tmp_path / "d.db", AccountsConfig(), providers=providers)
    weave.router._routing_service = SmartRoutingService(StaticCatalogAdapter(custom_offers=offers))
    return weave


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,provider", [(ProviderType.MODAL, "modal"), (ProviderType.LIGHTNING, "lightning"), (ProviderType.SKYPILOT, "runpod")])
async def test_no_wall_clock_or_latency_is_invented_compute_duration(tmp_path, kind, provider):
    fake = FakeProvider(provider_name=provider, kind=kind)
    weave = setup_weave(tmp_path, [offer(provider)], [fake])
    try:
        handle = await weave.deploy("fish-s2-pro", provider=provider, gpu_type="L4", wait_for_ready=False,
                                    custom_args={"allow_spot": False})
        record = await weave.lifecycle_service.get_record(handle.id)
        record.created_at = datetime.now(UTC) - timedelta(days=365)
        record.ready_at = record.created_at
        await weave.lifecycle_service.repository.save(record)
        report = await handle.usage()
        assert report.cost.estimated_cost_usd is None
        assert report.cost.measured_gpu_seconds is report.cost.actual_billed_cost_usd is None
        assert report.cost.compute_seconds is None
        assert report.pricing.price_per_hour_usd == 3.6
        scenario = (await handle.usage(compute_seconds=61)).cost
        assert scenario.duration_source == "caller_supplied"
        assert scenario.estimated_billable_seconds == 120
        assert scenario.estimated_cost_usd == pytest.approx(0.12)
        assert scenario.measured_gpu_seconds is scenario.actual_billed_cost_usd is None
        await handle.stop()
        stopped = await weave.get_usage(handle.id)
        assert stopped.cost.estimated_cost_usd is None
        assert stopped.resource_status == "retained_or_unknown"
    finally:
        await weave.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu,count,expected", [("L4", 1, 3.6), ("H100", 1, None), ("L4", 2, None)])
async def test_explicit_overrides_cannot_reuse_previous_quote(tmp_path, gpu, count, expected):
    fake = FakeProvider()
    weave = setup_weave(tmp_path, [offer()], [fake])
    try:
        await weave.deploy("fish-s2-pro", provider="modal", wait_for_ready=False)
        handle = await weave.deploy("fish-s2-pro", provider="modal", gpu_type=gpu, num_gpus=count, wait_for_ready=False)
        report = await handle.usage(compute_seconds=60)
        assert report.pricing.gpu_type == gpu and report.pricing.gpu_count == count
        assert report.pricing.price_per_hour_usd == expected
        if expected is None:
            assert report.cost.estimated_cost_usd is None
    finally:
        await weave.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_spot,spot,expected", [(True, 1.8, 1.8), (False, 1.8, 3.6), (True, None, None)])
async def test_spot_policy_never_falls_back_to_on_demand_rate(tmp_path, allow_spot, spot, expected):
    fake = FakeProvider(provider_name="runpod", kind=ProviderType.SKYPILOT)
    weave = setup_weave(tmp_path, [offer("runpod", spot_price_per_hour=spot)], [fake])
    try:
        handle = await weave.deploy("fish-s2-pro", provider="runpod", gpu_type="L4", wait_for_ready=False,
                                    custom_args={"allow_spot": allow_spot})
        report = await handle.usage(compute_seconds=60)
        assert report.pricing.price_per_hour_usd == expected
        assert report.pricing.price_kind == ("spot" if allow_spot else "on_demand")
        assert (report.cost.estimated_cost_usd is None) == (expected is None)
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_provider_failover_carries_its_own_gpu_quote(tmp_path, monkeypatch):
    monkeypatch.setattr("inferweave.services.routing_service.sys", SimpleNamespace(platform="linux"))
    first = FakeProvider().script("provision", fail(FailureKind.CAPACITY))
    second = FakeProvider(provider_name="lightning", kind=ProviderType.LIGHTNING)
    second_offer = offer("lightning", gpu="H100").model_copy(update={"price_per_hour": 7.2})
    weave = setup_weave(tmp_path, [offer(), second_offer], [first, second])
    try:
        handle = await weave.deploy("fish-s2-pro", provider="auto", wait_for_ready=False)
        report = await handle.usage(compute_seconds=60)
        assert handle.provider == report.pricing.provider == "lightning"
        assert report.pricing.gpu_type == "H100" and report.pricing.price_per_hour_usd == 7.2
        assert report.cost.estimated_cost_usd == pytest.approx(0.12)
        failed = next(r for r in await weave.list_records() if r.provider == "modal")
        assert failed.pricing.gpu_type == "L4" and failed.pricing.price_per_hour_usd == 3.6
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_dry_run_has_zero_compute_estimate_and_no_cloud_calls(tmp_path):
    fake = FakeProvider()
    weave = setup_weave(tmp_path, [offer()], [fake])
    try:
        handle = await weave.deploy("fish-s2-pro", provider="modal", dry_run=True)
        before = await handle.usage(compute_seconds=100)
        assert before.cost == CostEstimate(estimated_cost_usd=0, compute_seconds=0,
                                         estimated_billable_seconds=0, duration_source="dry_run")
        assert before.usage.requests == 0 and before.resource_status == "not_provisioned"
        await handle.stop()
        assert (await handle.usage()).cost == before.cost
        assert [c.op for c in fake.calls] == ["preflight"]
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_stop_down_uncertainty_and_confirmed_deletion_are_distinct(tmp_path):
    fake = FakeProvider(provider_name="runpod", kind=ProviderType.SKYPILOT)
    weave = setup_weave(tmp_path, [offer("runpod")], [fake])
    try:
        handle = await weave.deploy("fish-s2-pro", provider="runpod", gpu_type="L4", wait_for_ready=False,
                                    custom_args={"allow_spot": False})
        await handle.stop(action=AutostopAction.DOWN)
        report = await weave.get_usage(handle.id)
        assert report.resource_status == "down_acknowledged" and report.cost.estimated_cost_usd is None
        record = await weave.lifecycle_service.get_record(handle.id)
        record.needs_reconciliation = True
        record.mark_failed("Uncertain cleanup")
        await weave.lifecycle_service.repository.save(record)
        assert "deployment_outcome_uncertain" in (await handle.usage(compute_seconds=60)).cost.unavailable_reasons
        await weave.reconcile(min_age_seconds=0)
        confirmed = await handle.usage()
        assert confirmed.resource_status == "confirmed_absent"
        assert confirmed.cost.estimated_cost_usd is None
    finally:
        await weave.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
async def test_pricing_persistence_sanitizes_strings_and_excludes_offer_metadata(tmp_path, backend, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "SENTINEL-catalog-credential")
    repo = JsonDeploymentRepository(tmp_path / "d.json") if backend == "json" else SqliteDeploymentRepository(tmp_path / "d.db")
    record = DeploymentRecord(id="d", model="test", provider="modal", pricing=ComputePricing(
        provider="modal", gpu_count=1, gpu_type="L4", instance_type="SENTINEL-catalog-credential",
        region="https://example.test/?api_key=dummy-sensitive-key",
    ))
    await repo.save(record)
    text = (await repo.get("d")).model_dump_json()
    assert "SENTINEL-catalog-credential" not in text and "dummy-sensitive-key" not in text


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf")])
def test_invalid_scenario_duration_rejected(seconds):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        estimate_cost(None, seconds)


def test_zero_rate_zero_duration_and_configuration_total_not_per_gpu():
    quote = ComputePricing(provider="modal", gpu_count=2, price_per_hour_usd=3.6,
                           price_kind="on_demand", billing_granularity_seconds=1, source="catalog")
    assert estimate_cost(quote, 3600).estimated_cost_usd == 3.6  # no second GPU multiplier
    assert estimate_cost(quote, 0).estimated_cost_usd == 0
    assert estimate_cost(quote.model_copy(update={"price_per_hour_usd": 0}), 3600).estimated_cost_usd == 0


@pytest.mark.asyncio
async def test_pooled_secret_in_catalog_quote_is_sanitized_before_storage(tmp_path):
    from support import modal_pool

    secrets = modal_pool("a").secret_values()
    secret = next(iter(secrets))
    fake = FakeProvider()
    weave = make_weave(tmp_path / "d.db", modal_pool("a"), providers=[fake])
    quote = offer().model_copy(update={
        "instance_type": "example-" + secret,
        "metadata": {"credentials": {"password": "SENTINEL-nested-password"}},
    })
    weave.router._routing_service = SmartRoutingService(StaticCatalogAdapter(custom_offers=[quote]))
    try:
        handle = await weave.deploy("fish-s2-pro", provider="modal", wait_for_ready=False)
        record = await weave.lifecycle_service.repository.get(handle.id)
        assert secret not in record.model_dump_json()
        assert "SENTINEL-nested-password" not in record.model_dump_json()
        assert "[REDACTED]" in record.pricing.instance_type
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_multiple_gpu_quote_and_read_only_report_do_not_mutate_records(tmp_path):
    fake = FakeProvider()
    weave = setup_weave(tmp_path, [offer(count=2)], [fake])
    try:
        handle = await weave.deploy("fish-s2-pro", provider="modal", gpu_type="L4", num_gpus=2, wait_for_ready=False)
        before = await weave.lifecycle_service.repository.get(handle.id)
        calls = list(fake.calls)
        report = await handle.usage(compute_seconds=3600)
        assert report.pricing.gpu_count == 2 and report.cost.estimated_cost_usd == 3.6
        assert await weave.lifecycle_service.repository.get(handle.id) == before
        assert fake.calls == calls
    finally:
        await weave.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("regions,quotes,known", [
    ([], [offer(region="us-east-1")], False),
    (["eu-west-1"], [offer(region="us-east-1")], False),
    (["us-east-1"], [offer(region="us-east-1")], True),
    ([], [offer(), offer()], False),
])
async def test_region_mismatch_and_ambiguous_quotes_stay_unavailable(tmp_path, regions, quotes, known):
    weave = setup_weave(tmp_path, quotes, [FakeProvider()])
    try:
        handle = await weave.deploy("fish-s2-pro", provider="modal", gpu_type="L4", wait_for_ready=False,
                                    custom_args={"preferred_regions": regions})
        assert ((await handle.usage()).pricing.price_per_hour_usd is not None) == known
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_stop_failure_does_not_close_billing_or_erase_usage(tmp_path):
    fake = FakeProvider().script("stop", fail(FailureKind.TRANSIENT, resource_may_exist=True))
    weave = setup_weave(tmp_path, [offer()], [fake])
    try:
        handle = await weave.deploy("fish-s2-pro", provider="modal", wait_for_ready=False)
        with pytest.raises(ProviderOperationError):
            await handle.stop()
        report = await handle.usage(compute_seconds=60)
        assert report.resource_status == "unknown"
        assert report.cost.estimated_cost_usd is None and report.cost.actual_billed_cost_usd is None
        assert "deployment_outcome_uncertain" in report.cost.unavailable_reasons
        assert report.usage.requests == 0
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_skypilot_stop_with_retained_resource_is_not_free(tmp_path):
    class RetainingProvider(FakeProvider):
        async def stop(self, record, account, action=AutostopAction.STOP):
            if action == AutostopAction.DOWN:
                await super().stop(record, account, action)
            else:
                self._record("stop", record, account)
                self._check_owner("stop", record, account)

    fake = RetainingProvider(provider_name="runpod", kind=ProviderType.SKYPILOT)
    weave = setup_weave(tmp_path, [offer("runpod")], [fake])
    try:
        handle = await weave.deploy("fish-s2-pro", provider="runpod", gpu_type="L4", wait_for_ready=False,
                                    custom_args={"allow_spot": False})
        await handle.stop()
        record = await weave.lifecycle_service.repository.get(handle.id)
        assert await fake.resource_exists(record, weave.accounts.resolve_owner(record))
        report = await handle.usage()
        assert report.resource_status == "retained_or_unknown"
        assert report.cost.estimated_cost_usd is None
        await handle.stop(action=AutostopAction.DOWN)
        assert (await handle.usage()).resource_status == "down_acknowledged"
    finally:
        await weave.close()
