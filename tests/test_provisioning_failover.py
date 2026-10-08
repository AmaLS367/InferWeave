"""Provisioning through InferWeave.deploy: scheduling, classified failover, no duplicate resources."""

import asyncio
import logging
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
from fakes import (
    LIGHTNING_ENV,
    MODAL_ENV,
    MODEL,
    FakeClock,
    FakeProvider,
    accounts_config,
    contains_sentinel,
    crash,
    deploy,
    fail,
    hang,
    lightning_specs,
    make_weave,
    modal_env,
    modal_spec,
)

from inferweave.accounts import AccountManager, AccountsConfig, ProviderAccounts
from inferweave.adapters.lifecycle.memory_repository import InMemoryDeploymentRepository
from inferweave.core.exceptions import (
    AccountUnavailableError,
    NoAccountAvailableError,
    ProviderOperationError,
    ProvisioningUncertainError,
)
from inferweave.core.failures import FailureKind
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.registry.base import ModelRegistry
from inferweave.runtimes.templates import get_runtime_template
from inferweave.services.provisioning_service import (
    ProvisionCandidate,
    ProvisioningService,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def weave(tmp_path: Path, fake: FakeProvider, clock: FakeClock):
    return make_weave(
        tmp_path / "d.db", accounts_config(), dict(MODAL_ENV), [fake], clock=clock
    )


def _state(weave, account_id: str) -> str:
    return next(h for h in weave.account_health() if h.account_id == account_id).state


async def _failed_records(weave) -> list:
    return [r for r in await weave.list_records() if r.state == DeploymentState.FAILED]


async def test_new_deployments_round_robin_existing_keep_owner(
    weave, fake: FakeProvider
) -> None:
    deployments = [await deploy(weave) for _ in range(4)]
    assert [d.account for d in deployments] == ["a", "b", "a", "b"]
    for d in deployments:
        status = await weave.get_status(d.id)
        assert status.account == d.account
    assert [c.account_id for c in fake.ops("status")] == ["a", "b", "a", "b"]
    for d in deployments:
        record = await weave.lifecycle_service.get_record(d.id)
        assert record.account == d.account and not record.needs_reconciliation
        assert (
            record.resource.name == d.id
            and record.resource.resource_id == f"rid-{d.id}"
        )
    assert not fake.violations


async def test_rate_limit_fails_over_and_parks_first_account(
    weave, fake: FakeProvider, clock: FakeClock
) -> None:
    fake.script(
        "provision", fail(FailureKind.RATE_LIMIT, retry_after=120, status_code=429)
    )
    d = await deploy(weave)
    assert d.account == "b"
    assert fake.accounts_for("provision") == ["a", "b"]
    assert (
        fake.ops("exists", "stop") == []
    )  # definitive rejection: nothing to reconcile
    health = next(h for h in weave.account_health() if h.account_id == "a")
    assert health.state == "rate_limited"
    assert health.cooldown_until == clock.now() + timedelta(seconds=120)
    [failed] = await _failed_records(weave)
    assert failed.account == "a" and not failed.needs_reconciliation
    assert "rate_limit" in failed.error_message
    # While a cools down, every new deployment goes to b.
    assert (await deploy(weave)).account == "b"


async def test_auth_failure_revokes_and_fails_over(weave, fake: FakeProvider) -> None:
    fake.script("provision", fail(FailureKind.AUTH, status_code=401))
    assert (await deploy(weave)).account == "b"
    assert _state(weave, "a") == "revoked"
    assert [(await deploy(weave)).account for _ in range(2)] == ["b", "b"]


async def test_permission_denied_fails_over_without_revoking(
    weave, fake: FakeProvider
) -> None:
    fake.script("provision", fail(FailureKind.PERMISSION, status_code=403))
    assert (await deploy(weave)).account == "b"
    health = next(h for h in weave.account_health() if h.account_id == "a")
    assert health.state == "rate_limited" and health.consecutive_failures == 0
    assert health.state != "revoked"


async def test_quota_exhausted_fails_over(weave, fake: FakeProvider) -> None:
    fake.script("provision", fail(FailureKind.QUOTA, status_code=402))
    assert (await deploy(weave)).account == "b"
    assert _state(weave, "a") == "quota_exhausted"


async def test_transient_5xx_reconciles_absent_then_fails_over(
    weave, fake: FakeProvider
) -> None:
    fake.script("provision", fail(FailureKind.TRANSIENT, status_code=503))
    d = await deploy(weave)
    assert d.account == "b"
    assert [tuple(c[::2]) for c in fake.ops("provision", "exists", "stop")] == [
        ("provision", "a"),
        ("exists", "a"),
        ("provision", "b"),
    ]
    assert _state(weave, "a") == "cooldown"


async def test_timeout_after_create_destroys_under_same_account_first(
    weave, fake: FakeProvider
) -> None:
    fake.script(
        "provision", fail(FailureKind.TRANSIENT, create=True, message="timed out")
    )
    d = await deploy(weave)
    assert d.account == "b"
    sequence = [(c.op, c.account_id) for c in fake.ops("provision", "exists", "stop")]
    assert sequence == [
        ("provision", "a"),
        ("exists", "a"),
        ("stop", "a"),
        ("exists", "a"),
        ("provision", "b"),
    ]
    # Exactly one live (billed) resource remains, owned by the deployment's account.
    assert fake.live == {d.id: "b"}
    [failed] = await _failed_records(weave)
    assert failed.account == "a" and not failed.needs_reconciliation
    assert not fake.violations


async def test_reconcile_failure_stops_with_uncertain_error(
    weave, fake: FakeProvider
) -> None:
    fake.script("provision", fail(FailureKind.TRANSIENT, create=True))
    fake.script("exists", crash(ConnectionError("control plane down")))
    with pytest.raises(ProvisioningUncertainError) as info:
        await deploy(weave)
    assert fake.accounts_for("provision") == ["a"]  # no retry anywhere else
    record = await weave.lifecycle_service.get_record(info.value.deployment_id)
    assert record.needs_reconciliation is True
    assert record.account == "a" == info.value.account_id
    assert record.state == DeploymentState.FAILED
    assert fake.live == {
        record.id: "a"
    }  # the possibly billed resource is left for reconcile()
    assert "reconcile" in str(info.value)


async def test_uncertain_when_resource_survives_stop(weave, fake: FakeProvider) -> None:
    fake.script("provision", fail(FailureKind.TRANSIENT, create=True))
    fake.script("stop", crash(TimeoutError()))
    with pytest.raises(ProvisioningUncertainError):
        await deploy(weave)
    assert fake.accounts_for("provision") == ["a"]


async def test_invalid_request_never_fails_over(weave, fake: FakeProvider) -> None:
    fake.script("provision", fail(FailureKind.INVALID_REQUEST, status_code=422))
    with pytest.raises(ProviderOperationError) as info:
        await deploy(weave)
    assert info.value.kind is FailureKind.INVALID_REQUEST
    assert fake.accounts_for("provision") == ["a"]
    assert fake.ops("exists") == []
    assert _state(weave, "a") == "available"


async def test_capacity_fails_over_and_keeps_account_healthy(
    weave, fake: FakeProvider
) -> None:
    fake.script("provision", fail(FailureKind.CAPACITY))
    assert (await deploy(weave)).account == "b"
    assert [c.op for c in fake.ops("exists")] == [
        "exists"
    ]  # capacity may be ambiguous: checked
    health = next(h for h in weave.account_health() if h.account_id == "a")
    assert (health.state, health.consecutive_failures) == ("available", 0)
    assert (await deploy(weave)).account == "a"


async def test_all_accounts_exhausted_lists_attempts(weave, fake: FakeProvider) -> None:
    fake.script(
        "provision",
        fail(FailureKind.RATE_LIMIT, retry_after=90),
        fail(FailureKind.RATE_LIMIT, retry_after=40),
    )
    with pytest.raises(NoAccountAvailableError) as info:
        await deploy(weave)
    assert info.value.attempts == [
        ("modal", "a", "rate_limit"),
        ("modal", "b", "rate_limit"),
    ]
    assert info.value.retry_after == pytest.approx(40)
    assert "modal/a: rate_limit" in str(info.value) and "modal/b: rate_limit" in str(
        info.value
    )
    assert isinstance(info.value.__cause__, ProviderOperationError)
    assert len(fake.ops("provision")) == 2


async def test_max_attempts_bounds_provision_calls(tmp_path: Path) -> None:
    env = {**MODAL_ENV, **modal_env("c")}
    config = AccountsConfig(
        {
            "modal": ProviderAccounts(
                accounts=[modal_spec("a"), modal_spec("b"), modal_spec("c")]
            )
        },
        max_attempts=2,
    )
    fake = FakeProvider().script("provision", *[fail(FailureKind.TRANSIENT)] * 3)
    weave = make_weave(tmp_path / "d.db", config, env, [fake])
    with pytest.raises(NoAccountAvailableError) as info:
        await deploy(weave)
    assert len(fake.ops("provision")) == 2
    assert [a for _, a, _ in info.value.attempts] == ["a", "b"]


async def test_single_ambient_account_reraises_original_error(tmp_path: Path) -> None:
    fake = FakeProvider().script(
        "provision", fail(FailureKind.RATE_LIMIT, retry_after=3, message="slow down")
    )
    weave = make_weave(tmp_path / "d.db", AccountsConfig(), {}, [fake])
    with pytest.raises(ProviderOperationError) as info:
        await deploy(weave)
    assert type(info.value) is ProviderOperationError
    assert info.value.kind is FailureKind.RATE_LIMIT and str(info.value) == "slow down"
    assert fake.accounts_for("provision") == ["ambient"]


async def test_ambient_success_records_ambient_owner(tmp_path: Path) -> None:
    fake = FakeProvider()
    weave = make_weave(tmp_path / "d.db", AccountsConfig(), {}, [fake])
    d = await deploy(weave)
    assert d.account == "ambient"
    await weave.stop(d.id)
    assert fake.accounts_for("stop") == ["ambient"]


async def test_unclassified_sdk_error_text_never_leaks(
    weave, fake: FakeProvider, caplog: pytest.LogCaptureFixture
) -> None:
    secret = MODAL_ENV["IW_MODAL_A_TOKEN_SECRET"]
    fake.script(
        "provision",
        crash(RuntimeError(f"401 for token_secret={secret}")),
        crash(
            ValueError(
                f"bad header Modal-Secret: {MODAL_ENV['IW_MODAL_B_PROXY_SECRET']}"
            )
        ),
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(ProvisioningUncertainError) as info:
        await deploy(weave)
    err = info.value
    rendered = [str(err), repr(err), "".join(traceback.format_exception(err))]
    assert fake.accounts_for("provision") == ["a"]
    for text in rendered:
        assert not contains_sentinel(text), text
    for record in caplog.records:
        assert not contains_sentinel(record.getMessage())
        assert not contains_sentinel(str(record.exc_text or ""))
    for record in await weave.list_records():
        assert not contains_sentinel(record.model_dump_json())


async def test_unclassified_error_on_single_account_is_wrapped(tmp_path: Path) -> None:
    fake = FakeProvider().script("provision", crash(KeyError("SENTINEL-x")))
    weave = make_weave(tmp_path / "d.db", AccountsConfig(), {}, [fake])
    with pytest.raises(ProvisioningUncertainError) as info:
        await deploy(weave)
    assert (await weave.list_records())[0].creation_may_continue
    assert "SENTINEL" not in "".join(traceback.format_exception(info.value))


async def test_pinned_account_uses_only_that_account(weave, fake: FakeProvider) -> None:
    assert [(await deploy(weave, account="b")).account for _ in range(3)] == [
        "b",
        "b",
        "b",
    ]
    fake.script("provision", fail(FailureKind.RATE_LIMIT))
    with pytest.raises(ProviderOperationError) as info:
        await deploy(weave, account="a")
    assert info.value.kind is FailureKind.RATE_LIMIT
    assert fake.accounts_for("provision") == ["b", "b", "b", "a"]


async def test_pinned_unknown_account_and_auto_provider(
    weave, fake: FakeProvider
) -> None:
    with pytest.raises(AccountUnavailableError, match="'zzz'"):
        await deploy(weave, account="zzz")
    with pytest.raises(ValueError, match="explicit provider"):
        await deploy(weave, provider="auto", account="a")
    assert fake.ops("provision") == []


async def test_pinned_transient_failure_does_not_fail_over(
    weave, fake: FakeProvider
) -> None:
    fake.script("provision", fail(FailureKind.TRANSIENT, create=True))
    with pytest.raises(ProviderOperationError):
        await deploy(weave, account="b")
    assert fake.accounts_for("provision") == ["b"]
    assert [(c.op, c.account_id) for c in fake.ops("exists", "stop")] == [
        ("exists", "b"),
        ("stop", "b"),
        ("exists", "b"),
    ]
    assert fake.live == {}


def _candidate(provider: FakeProvider, name: str) -> ProvisionCandidate:
    profile = ModelRegistry().get(MODEL)
    request = DeploymentRequest(model=MODEL, provider=name, gpu_type="L4")
    runtime = get_runtime_template(profile.default_runtime).render(profile, request)
    return ProvisionCandidate(provider=provider, request=request, runtime=runtime)


async def test_cross_provider_failover_after_first_provider_exhausted() -> None:
    modal = FakeProvider().script(
        "provision", fail(FailureKind.CAPACITY), fail(FailureKind.CAPACITY)
    )
    lightning = FakeProvider(provider_name="lightning", kind=ProviderType.LIGHTNING)
    env = {**MODAL_ENV, **LIGHTNING_ENV}
    config = AccountsConfig(
        {
            "modal": ProviderAccounts(accounts=[modal_spec("a"), modal_spec("b")]),
            "lightning": ProviderAccounts(accounts=lightning_specs()),
        },
        max_attempts=4,
    )
    service = ProvisioningService(
        AccountManager(config, environ=env), InMemoryDeploymentRepository()
    )
    profile = ModelRegistry().get(MODEL)
    record, provider = await service.provision(
        [_candidate(modal, "modal"), _candidate(lightning, "lightning")], profile
    )
    assert provider is lightning
    assert (record.provider, record.account) == ("lightning", "a")
    assert modal.accounts_for("provision") == ["a", "b"]
    assert lightning.accounts_for("provision") == ["a"]


async def test_cross_provider_budget_is_shared() -> None:
    modal = FakeProvider().script(
        "provision", fail(FailureKind.CAPACITY), fail(FailureKind.CAPACITY)
    )
    lightning = FakeProvider(provider_name="lightning", kind=ProviderType.LIGHTNING)
    config = AccountsConfig(
        {"modal": ProviderAccounts(accounts=[modal_spec("a"), modal_spec("b")])},
        max_attempts=2,
    )
    service = ProvisioningService(
        AccountManager(config, environ=dict(MODAL_ENV)), InMemoryDeploymentRepository()
    )
    with pytest.raises(NoAccountAvailableError) as info:
        await service.provision(
            [_candidate(modal, "modal"), _candidate(lightning, "lightning")],
            ModelRegistry().get(MODEL),
        )
    assert len(info.value.attempts) == 2
    assert lightning.ops("provision") == []


async def test_auto_deploy_falls_over_to_next_routed_provider(tmp_path: Path) -> None:
    modal = FakeProvider().script(
        "provision", fail(FailureKind.QUOTA), fail(FailureKind.QUOTA)
    )
    lightning = FakeProvider(provider_name="lightning", kind=ProviderType.LIGHTNING)
    weave = make_weave(
        tmp_path / "d.db",
        accounts_config(max_attempts=5),
        dict(MODAL_ENV),
        [modal, lightning],
    )

    async def candidates(provider_name, profile, strategy, request):
        alternative = request.model_copy(deep=True, update={"provider": "lightning"})
        return [(modal, request), (lightning, alternative)]

    weave.router.acandidates = candidates  # type: ignore[method-assign]
    d = await deploy(weave, provider="auto")
    assert (d.provider, d.account) == ("lightning", "ambient")
    assert modal.accounts_for("provision") == ["a", "b"]
    await weave.stop(d.id)
    assert lightning.accounts_for("stop") == ["ambient"]


async def test_cancellation_mid_provision_cleans_up(weave, fake: FakeProvider) -> None:
    fake.script("provision", hang(create=True))
    task = asyncio.create_task(deploy(weave))
    await asyncio.wait_for(fake.entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.live == {}
    assert [(c.op, c.account_id) for c in fake.ops("exists", "stop")] == [
        ("exists", "a"),
        ("stop", "a"),
        ("exists", "a"),
    ]
    [record] = await weave.list_records()
    assert record.state == DeploymentState.FAILED and not record.needs_reconciliation
    assert record.account == "a"
    assert next(h for h in weave.account_health() if h.account_id == "a").in_flight == 0


async def test_cancellation_with_unknown_outcome_flags_record(
    weave, fake: FakeProvider
) -> None:
    fake.script("provision", hang(create=True))
    fake.script("exists", crash(ConnectionError()))
    task = asyncio.create_task(deploy(weave))
    await asyncio.wait_for(fake.entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [record] = await weave.list_records()
    assert record.needs_reconciliation and record.account == "a"
    assert record.state == DeploymentState.FAILED
    assert "reconcile" in record.error_message


async def test_write_ahead_record_exists_during_provision(
    weave, fake: FakeProvider
) -> None:
    seen = []

    async def check(record, account) -> None:
        stored = await weave.lifecycle_service.get_record(record.id)
        seen.append(stored)

    fake.before_provision = check
    d = await deploy(weave)
    [stored] = seen
    assert stored is not None
    assert stored.needs_reconciliation is True
    assert stored.account == d.account == "a"
    assert stored.state == DeploymentState.PROVISIONING
    assert stored.resource is not None and stored.resource.name == d.id
    final = await weave.lifecycle_service.get_record(d.id)
    assert (
        final.needs_reconciliation is False and final.state == DeploymentState.STARTING
    )


async def test_dry_run_uses_no_account(weave, fake: FakeProvider) -> None:
    d = await deploy(weave, dry_run=True)
    assert d.account is None
    assert [(c.op, c.account_id) for c in fake.calls] == [("preflight", "ambient")]
    record = await weave.lifecycle_service.get_record(d.id)
    assert record.account is None and record.is_dry_run
    assert all(h.total_leases == 0 for h in weave.account_health())


async def test_success_is_reported_to_credweave(weave, fake: FakeProvider) -> None:
    await deploy(weave)
    health = {h.account_id: h for h in weave.account_health()}
    assert health["a"].total_leases == 1 and health["a"].in_flight == 0
    assert health["b"].total_leases == 0
