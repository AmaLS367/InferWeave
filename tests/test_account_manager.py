"""AccountManager: CredWeave scheduling, outcome mapping, owner resolution and hot reload."""

import json
import logging
import os
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pytest
from fakes import (
    MODAL_ENV,
    SENTINELS,
    FakeClock,
    contains_sentinel,
    modal_env,
    modal_spec,
    modal_specs,
)

from inferweave.accounts import (
    AccountManager,
    AccountsConfig,
    AccountSpec,
    ProviderAccount,
    ProviderAccounts,
)
from inferweave.core.exceptions import (
    AccountUnavailableError,
    NoAccountAvailableError,
    ProviderOperationError,
)
from inferweave.core.failures import FailureKind

pytestmark = pytest.mark.asyncio


def _manager(
    *specs: AccountSpec,
    environ: dict[str, str] | None = None,
    clock: FakeClock | None = None,
    **pool: object,
) -> AccountManager:
    config = AccountsConfig(
        {"modal": ProviderAccounts(accounts=list(specs or modal_specs()), **pool)}  # type: ignore[arg-type]
    )
    return AccountManager(
        config,
        clock=clock or FakeClock(),
        environ=environ if environ is not None else dict(MODAL_ENV),
    )


def _err(kind: FailureKind, retry_after: float | None = None) -> ProviderOperationError:
    return ProviderOperationError("scripted", kind, retry_after=retry_after)


async def _pick(manager: AccountManager, **kwargs: object) -> str:
    lease = await manager.acquire("modal", **kwargs)  # type: ignore[arg-type]
    await lease.report_success()
    return lease.account_id


def _health(manager: AccountManager, account_id: str):
    return next(h for h in manager.health() if h.account_id == account_id)


# --- strategies -----------------------------------------------------------------------------------


async def test_round_robin_alternates() -> None:
    manager = _manager()
    assert [await _pick(manager) for _ in range(5)] == ["a", "b", "a", "b", "a"]


async def test_weighted_3_to_1_over_8_picks() -> None:
    env = {**MODAL_ENV}
    manager = _manager(
        modal_spec("a", weight=3),
        modal_spec("b", weight=1),
        environ=env,
        strategy="weighted",
    )
    picks = [await _pick(manager) for _ in range(8)]
    assert Counter(picks) == {"a": 6, "b": 2}
    # Smooth weighted round robin interleaves instead of bursting.
    assert picks[:4].count("b") == 1


async def test_failover_prefers_priority_and_falls_back() -> None:
    clock = FakeClock()
    manager = _manager(
        modal_spec("a", priority=2),
        modal_spec("b", priority=1),
        clock=clock,
        strategy="failover",
    )
    assert [await _pick(manager) for _ in range(3)] == ["b", "b", "b"]
    lease = await manager.acquire("modal")
    await lease.report_failure(_err(FailureKind.RATE_LIMIT, retry_after=30))
    assert [await _pick(manager) for _ in range(2)] == ["a", "a"]
    clock.advance(31)
    assert await _pick(manager) == "b"


async def test_priority_alias_is_failover() -> None:
    manager = _manager(
        modal_spec("a", priority=5), modal_spec("b", priority=0), strategy="priority"
    )
    assert await _pick(manager) == "b"


async def test_least_used_picks_lowest_total_leases() -> None:
    manager = _manager(strategy="least_used")
    assert await _pick(manager, pin="a") == "a"
    assert await _pick(manager, pin="a") == "a"
    assert [await _pick(manager) for _ in range(3)] == ["b", "b", "a"]


# --- exclusion & pinning -------------------------------------------------------------------------


async def test_exclusion_skips_accounts() -> None:
    manager = _manager()
    assert {await _pick(manager, exclude={"a"}) for _ in range(3)} == {"b"}
    with pytest.raises(NoAccountAvailableError):
        await manager.acquire("modal", exclude={"a", "b"})


async def test_pin_selects_exactly_one_account() -> None:
    clock = FakeClock()
    manager = _manager(clock=clock)
    assert [await _pick(manager, pin="b") for _ in range(3)] == ["b", "b", "b"]
    with pytest.raises(AccountUnavailableError, match="'zzz'"):
        await manager.acquire("modal", pin="zzz")
    lease = await manager.acquire("modal", pin="b")
    await lease.report_failure(_err(FailureKind.RATE_LIMIT, retry_after=10))
    # A pinned account that is cooling down is unavailable; no substitute is returned.
    with pytest.raises(NoAccountAvailableError):
        await manager.acquire("modal", pin="b")


async def test_unpooled_provider_uses_ambient_account() -> None:
    manager = _manager()
    lease = await manager.acquire("lightning")
    assert lease.account.is_ambient and lease.account_id == "ambient"
    await lease.report_failure(_err(FailureKind.AUTH))  # no pool: a no-op
    assert (await manager.acquire("lightning", pin="ambient")).account.is_ambient
    with pytest.raises(AccountUnavailableError):
        await manager.acquire("lightning", pin="team")
    with pytest.raises(NoAccountAvailableError, match="ambient"):
        await manager.acquire("lightning", exclude={"ambient"})
    assert manager.resolve("lightning", None).is_ambient
    assert manager.resolve("modal", "ambient").is_ambient
    assert manager.account_ids("lightning") == ["ambient"]
    assert manager.account_ids("MODAL") == ["a", "b"]
    assert manager.has_pool("Modal") and not manager.has_pool("lightning")


# --- outcome mapping -----------------------------------------------------------------------------


async def test_rate_limit_cools_down_for_retry_after() -> None:
    clock = FakeClock()
    manager = _manager(clock=clock)
    lease = await manager.acquire("modal", pin="a")
    await lease.report_failure(_err(FailureKind.RATE_LIMIT, retry_after=30))
    health = _health(manager, "a")
    assert health.state == "rate_limited"
    assert health.cooldown_until == clock.now() + timedelta(seconds=30)
    assert health.consecutive_failures == 0
    assert {await _pick(manager) for _ in range(3)} == {"b"}
    with pytest.raises(NoAccountAvailableError) as info:
        await manager.acquire("modal", exclude={"b"})
    assert info.value.retry_after == pytest.approx(30)
    clock.advance(10)
    with pytest.raises(NoAccountAvailableError) as info:
        await manager.acquire("modal", exclude={"b"})
    assert info.value.retry_after == pytest.approx(20)
    clock.advance(21)
    assert await _pick(manager, exclude={"b"}) == "a"


async def test_no_account_retry_after_is_earliest_recovery() -> None:
    clock = FakeClock()
    manager = _manager(clock=clock)
    for account_id, delay in (("a", 50), ("b", 20)):
        lease = await manager.acquire("modal", pin=account_id)
        await lease.report_failure(_err(FailureKind.RATE_LIMIT, retry_after=delay))
    with pytest.raises(NoAccountAvailableError) as info:
        await manager.acquire("modal")
    assert info.value.retry_after == pytest.approx(20)


async def test_auth_revokes_until_authorize_or_reset() -> None:
    clock = FakeClock()
    manager = _manager(clock=clock)
    lease = await manager.acquire("modal", pin="a")
    await lease.report_failure(_err(FailureKind.AUTH))
    assert _health(manager, "a").state == "revoked"
    clock.advance(10 * 24 * 3600)  # revocation never times out
    assert {await _pick(manager) for _ in range(4)} == {"b"}
    await manager.authorize("modal", "a")
    assert _health(manager, "a").state == "available"
    assert "a" in {await _pick(manager) for _ in range(2)}

    lease = await manager.acquire("modal", pin="a")
    await lease.report_failure(_err(FailureKind.AUTH))
    assert _health(manager, "a").state == "revoked"
    await manager.reset("modal", "a")
    assert _health(manager, "a").state == "available"


async def test_permission_parks_without_health_damage() -> None:
    clock = FakeClock()
    manager = _manager(clock=clock, permission_cooldown_seconds=900)
    lease = await manager.acquire("modal", pin="a")
    await lease.report_failure(_err(FailureKind.PERMISSION))
    health = _health(manager, "a")
    assert health.state == "rate_limited"
    assert health.consecutive_failures == 0
    assert health.cooldown_until == clock.now() + timedelta(seconds=900)
    clock.advance(901)
    assert await _pick(manager, pin="a") == "a"


async def test_quota_uses_configured_cooldown_without_hint() -> None:
    clock = FakeClock()
    manager = _manager(clock=clock, quota_cooldown_seconds=3600)
    lease = await manager.acquire("modal", pin="a")
    await lease.report_failure(_err(FailureKind.QUOTA))
    health = _health(manager, "a")
    assert health.state == "quota_exhausted"
    assert health.cooldown_until == clock.now() + timedelta(seconds=3600)
    assert health.consecutive_failures == 0

    lease = await manager.acquire("modal", pin="b")
    await lease.report_failure(_err(FailureKind.QUOTA, retry_after=60))
    assert _health(manager, "b").cooldown_until == clock.now() + timedelta(seconds=60)
    clock.advance(61)
    assert await _pick(manager) == "b"


@pytest.mark.parametrize("kind", [FailureKind.CAPACITY, FailureKind.INVALID_REQUEST])
async def test_capacity_and_invalid_request_keep_account_available(
    kind: FailureKind,
) -> None:
    manager = _manager()
    lease = await manager.acquire("modal")
    assert lease.account_id == "a"
    await lease.report_failure(_err(kind))
    health = _health(manager, "a")
    assert (health.state, health.consecutive_failures, health.cooldown_until) == (
        "available",
        0,
        None,
    )
    assert [await _pick(manager) for _ in range(2)] == ["b", "a"]


async def test_transient_escalates_to_unhealthy() -> None:
    clock = FakeClock()
    manager = _manager(clock=clock, cooldown_seconds=10, max_consecutive_failures=2)
    lease = await manager.acquire("modal", pin="a")
    await lease.report_failure(_err(FailureKind.TRANSIENT))
    health = _health(manager, "a")
    assert (health.state, health.consecutive_failures) == ("cooldown", 1)
    assert health.cooldown_until == clock.now() + timedelta(seconds=10)
    clock.advance(11)
    lease = await manager.acquire("modal", pin="a")
    await lease.report_failure(_err(FailureKind.TRANSIENT))
    assert _health(manager, "a").state == "unhealthy"
    clock.advance(10_000)
    with pytest.raises(NoAccountAvailableError):
        await manager.acquire("modal", pin="a")


async def test_success_resets_transient_counter() -> None:
    clock = FakeClock()
    manager = _manager(clock=clock, cooldown_seconds=1, max_consecutive_failures=2)
    lease = await manager.acquire("modal", pin="a")
    await lease.report_failure(_err(FailureKind.TRANSIENT))
    clock.advance(2)
    await _pick(manager, pin="a")
    assert _health(manager, "a").consecutive_failures == 0


async def test_lease_reports_only_once() -> None:
    manager = _manager()
    lease = await manager.acquire("modal", pin="a")
    await lease.report_success()
    await lease.report_failure(_err(FailureKind.AUTH))  # ignored
    assert _health(manager, "a").state == "available"
    assert _health(manager, "a").in_flight == 0


async def test_outcome_for_maps_every_kind() -> None:
    manager = _manager(quota_cooldown_seconds=77, permission_cooldown_seconds=88)
    rl = manager.outcome_for(
        "modal",
        ProviderOperationError(
            "x", FailureKind.RATE_LIMIT, status_code=429, retry_after=5
        ),
    )
    assert (str(rl.type), rl.retry_after) == ("rate_limited", 5)
    assert rl.metadata["status_code"] == "429"
    assert (
        str(manager.outcome_for("modal", _err(FailureKind.AUTH)).type) == "auth_failed"
    )
    assert manager.outcome_for("modal", _err(FailureKind.QUOTA)).retry_after == 77
    assert manager.outcome_for("modal", _err(FailureKind.PERMISSION)).retry_after == 88
    assert (
        str(manager.outcome_for("modal", _err(FailureKind.TRANSIENT)).type)
        == "transient_error"
    )
    for kind in (FailureKind.CAPACITY, FailureKind.INVALID_REQUEST):
        assert str(manager.outcome_for("modal", _err(kind)).type) == "success"
    # Unpooled provider: no configured cooldowns.
    assert manager.outcome_for("runpod", _err(FailureKind.QUOTA)).retry_after is None


# --- dynamic accounts ----------------------------------------------------------------------------


async def test_env_rotation_keeps_id_and_serves_new_secret() -> None:
    env = dict(MODAL_ENV)
    manager = _manager(environ=env)
    before = manager.resolve("modal", "a")
    assert before.secret("token_secret") == "SENTINEL-modal-a-token-secret"
    env["IW_MODAL_A_TOKEN_SECRET"] = "SENTINEL-modal-a-rotated"
    after = manager.resolve("modal", "a")
    assert after.id == "a"
    assert after.secret("token_secret") == "SENTINEL-modal-a-rotated"
    assert after.secret_fingerprint != before.secret_fingerprint
    lease = await manager.acquire("modal", pin="a")
    assert lease.account.secret("token_secret") == "SENTINEL-modal-a-rotated"
    await lease.report_success()
    assert "SENTINEL-modal-a-rotated" in manager.secret_values()
    assert "SENTINEL-modal-a-token-secret" not in manager.secret_values()


async def test_account_added_at_runtime_gets_leases() -> None:
    env = dict(MODAL_ENV)
    manager = _manager(modal_spec("a"), modal_spec("b"), modal_spec("c"), environ=env)
    assert manager.account_ids("modal") == ["a", "b"]
    assert {await _pick(manager) for _ in range(4)} == {"a", "b"}
    env.update(modal_env("c"))
    assert manager.account_ids("modal") == ["a", "b", "c"]
    assert "c" in {await _pick(manager) for _ in range(3)}
    assert (
        manager.resolve("modal", "c").secret("token_id") == "SENTINEL-modal-c-token-id"
    )


async def test_removed_account_is_unavailable_and_never_leased(
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = dict(MODAL_ENV)
    manager = _manager(environ=env)
    assert manager.resolve("modal", "b").id == "b"
    del env["IW_MODAL_B_TOKEN_ID"]
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(AccountUnavailableError, match="'b'.*deployment 'dep-1'"):
            manager.resolve("modal", "b", "dep-1")
        assert {await _pick(manager) for _ in range(6)} == {"a"}
        with pytest.raises(AccountUnavailableError):
            await manager.acquire("modal", pin="b")
    assert "IW_MODAL_B_TOKEN_ID" in caplog.text  # the variable name, never a value
    assert not contains_sentinel(caplog.text)


def _write_json(path: Path, credentials: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"credentials": credentials}), encoding="utf-8")
    os.replace(temporary, path)


def _runpod_cred(account_id: str, key: str) -> dict:
    return {"id": account_id, "secrets": {"api_key": key}}


def _json_manager(path: Path, clock: FakeClock | None = None) -> AccountManager:
    config = AccountsConfig({"runpod": ProviderAccounts(credentials_file=path)})
    return AccountManager(config, clock=clock or FakeClock(), environ={})


async def test_json_credentials_file_hot_reload(tmp_path: Path) -> None:
    path = tmp_path / "runpod.json"
    _write_json(
        path,
        [_runpod_cred("r1", "SENTINEL-r1-v1"), _runpod_cred("r2", "SENTINEL-r2-v1")],
    )
    manager = _json_manager(path)
    assert manager.account_ids("runpod") == ["r1", "r2"]

    # add
    _write_json(
        path,
        [
            _runpod_cred("r1", "SENTINEL-r1-v1"),
            _runpod_cred("r2", "SENTINEL-r2-v1"),
            _runpod_cred("r3", "SENTINEL-r3-v1"),
        ],
    )
    assert manager.account_ids("runpod") == ["r1", "r2", "r3"]
    picks = set()
    for _ in range(3):
        lease = await manager.acquire("runpod")
        picks.add(lease.account_id)
        await lease.report_success()
    assert picks == {"r1", "r2", "r3"}

    # rotate r1, remove r2
    _write_json(
        path,
        [_runpod_cred("r1", "SENTINEL-r1-v2"), _runpod_cred("r3", "SENTINEL-r3-v1")],
    )
    assert manager.resolve("runpod", "r1").secret("api_key") == "SENTINEL-r1-v2"
    with pytest.raises(AccountUnavailableError):
        manager.resolve("runpod", "r2")
    for _ in range(4):
        lease = await manager.acquire("runpod")
        assert lease.account_id in {"r1", "r3"}
        if lease.account_id == "r1":
            assert lease.account.secret("api_key") == "SENTINEL-r1-v2"
        await lease.report_success()


async def test_invalid_json_credential_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "modal.json"
    _write_json(
        path,
        [
            {
                "id": "good",
                "secrets": {
                    "token_id": "SENTINEL-good-id",
                    "token_secret": "SENTINEL-good-s",
                },
            },
            {"id": "half", "secrets": {"token_id": "SENTINEL-half-id"}},
            {
                "id": "odd-meta",
                "secrets": {
                    "token_id": "SENTINEL-odd-id",
                    "token_secret": "SENTINEL-odd-s",
                },
                "metadata": {"region": "eu"},
            },
        ],
    )
    config = AccountsConfig({"modal": ProviderAccounts(credentials_file=path)})
    with caplog.at_level(logging.WARNING):
        manager = AccountManager(config, environ={})
        assert manager.account_ids("modal") == ["good"]
    assert "'half'" in caplog.text and "'odd-meta'" in caplog.text
    assert "SENTINEL" not in caplog.text
    with pytest.raises(AccountUnavailableError):
        manager.resolve("modal", "half")


async def test_duplicate_ids_across_sources_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "modal.json"
    _write_json(
        path, [{"id": "a", "secrets": {"token_id": "x1", "token_secret": "x2"}}]
    )
    config = AccountsConfig(
        {"modal": ProviderAccounts(accounts=[modal_spec("a")], credentials_file=path)}
    )
    with pytest.raises(Exception, match="more than once"):
        AccountManager(config, environ=dict(MODAL_ENV))


# --- nonsecret surface ---------------------------------------------------------------------------


async def test_health_and_reprs_are_nonsecret() -> None:
    manager = _manager()
    lease = await manager.acquire("modal")
    in_flight = _health(manager, lease.account_id)
    assert in_flight.in_flight == 1 and in_flight.total_leases == 1
    await lease.report_success()
    report = manager.health()
    assert [(h.provider, h.account_id, h.state) for h in report] == [
        ("modal", "a", "available"),
        ("modal", "b", "available"),
    ]
    for text in (
        repr(report),
        repr(manager),
        repr(lease),
        str(lease.account),
        repr(manager.resolve("modal", "b")),
    ):
        assert not contains_sentinel(text), text


async def test_secret_values_and_redact() -> None:
    manager = _manager()
    values = set(manager.secret_values())
    assert values == set(MODAL_ENV.values())
    text = "token SENTINEL-modal-a-token-secret and SENTINEL-modal-b-proxy-id plus extra-1234"
    redacted = manager.redact(text, extra=["extra-1234"])
    assert redacted == "token *** and *** plus ***"
    assert not contains_sentinel(redacted, SENTINELS)


async def test_provider_account_accessors() -> None:
    manager = _manager()
    account = manager.resolve("modal", "a")
    assert not account.is_ambient
    assert account.secret("proxy_token_id") == "SENTINEL-modal-a-proxy-id"
    assert account.optional_secret("missing") is None and not account.has_secret(
        "missing"
    )
    assert account.get_metadata("environment", "default") == "default"
    with pytest.raises(Exception, match="has no 'missing' secret") as info:
        account.secret("missing")
    assert not contains_sentinel(str(info.value))
    ambient = ProviderAccount.ambient("modal")
    assert (
        ambient.is_ambient
        and ambient.secret_values() == ()
        and ambient.secret_fingerprint is None
    )
