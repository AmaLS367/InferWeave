"""Shared fakes for the multi-account (CredWeave) test suites.

* :class:`FakeProvider` is a scriptable, stateless ``ComputeProvider`` that simulates one cloud.
  It remembers which account created every resource and flags any later status/stop/exists call
  made under a different account.
* :func:`make_weave` builds an offline ``InferWeave`` "process" over a SQLite database.
* ``*_ENV`` dicts hold sentinel secrets for two accounts per pooled provider; build matching
  account specs with :func:`modal_specs`, :func:`lightning_specs`, :func:`runpod_specs` and
  :func:`vast_specs` (or :func:`accounts_config` for a ready ``AccountsConfig``).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

from inferweave import InferWeave
from inferweave.accounts import (
    AccountManager,
    AccountsConfig,
    AccountSpec,
    ProviderAccount,
    ProviderAccounts,
    lightning_account,
    modal_account,
    runpod_account,
    vast_account,
)
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
from inferweave.core.exceptions import ProviderOperationError
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord, ResourceRef
from inferweave.domain.lifecycle import AutostopAction
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.ports.auth import EndpointAuthPort
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.providers.base import ComputeProvider, ProvisionResult, ResourceStatus
from inferweave.providers.router import ProviderRouter
from inferweave.runtimes.base import RuntimeSpec
from inferweave.services.healthcheck_service import HealthcheckService
from inferweave.services.lifecycle_service import LifecycleService

MODEL = "fish-s2-pro"

# --------------------------------------------------------------------------------------------
# Sentinel secrets
# --------------------------------------------------------------------------------------------


def modal_env(account_id: str) -> dict[str, str]:
    key = account_id.upper().replace("-", "_")
    return {
        f"IW_MODAL_{key}_TOKEN_ID": f"SENTINEL-modal-{account_id}-token-id",
        f"IW_MODAL_{key}_TOKEN_SECRET": f"SENTINEL-modal-{account_id}-token-secret",
        f"IW_MODAL_{key}_PROXY_ID": f"SENTINEL-modal-{account_id}-proxy-id",
        f"IW_MODAL_{key}_PROXY_SECRET": f"SENTINEL-modal-{account_id}-proxy-secret",
    }


def lightning_env(account_id: str) -> dict[str, str]:
    key = account_id.upper().replace("-", "_")
    return {
        f"IW_LIGHTNING_{key}_USER_ID": f"SENTINEL-lightning-{account_id}-user-id",
        f"IW_LIGHTNING_{key}_API_KEY": f"SENTINEL-lightning-{account_id}-api-key",
    }


def api_key_env(provider: str, account_id: str) -> dict[str, str]:
    key = account_id.upper().replace("-", "_")
    return {
        f"IW_{provider.upper()}_{key}_API_KEY": f"SENTINEL-{provider}-{account_id}-api-key"
    }


MODAL_ENV: dict[str, str] = {**modal_env("a"), **modal_env("b")}
LIGHTNING_ENV: dict[str, str] = {**lightning_env("a"), **lightning_env("b")}
RUNPOD_ENV: dict[str, str] = {
    **api_key_env("runpod", "a"),
    **api_key_env("runpod", "b"),
}
VAST_ENV: dict[str, str] = {**api_key_env("vast", "a"), **api_key_env("vast", "b")}
ALL_ENV: dict[str, str] = {**MODAL_ENV, **LIGHTNING_ENV, **RUNPOD_ENV, **VAST_ENV}
SENTINELS: tuple[str, ...] = tuple(ALL_ENV.values())


def modal_spec(account_id: str, *, proxy: bool = True, **kwargs: Any) -> AccountSpec:
    key = account_id.upper().replace("-", "_")
    return modal_account(
        account_id,
        token_id_env=f"IW_MODAL_{key}_TOKEN_ID",
        token_secret_env=f"IW_MODAL_{key}_TOKEN_SECRET",
        proxy_token_id_env=f"IW_MODAL_{key}_PROXY_ID" if proxy else None,
        proxy_token_secret_env=f"IW_MODAL_{key}_PROXY_SECRET" if proxy else None,
        **kwargs,
    )


def lightning_spec(
    account_id: str, *, teamspace: str | None = "org/ts", **kwargs: Any
) -> AccountSpec:
    key = account_id.upper().replace("-", "_")
    return lightning_account(
        account_id,
        user_id_env=f"IW_LIGHTNING_{key}_USER_ID",
        api_key_env=f"IW_LIGHTNING_{key}_API_KEY",
        teamspace=teamspace,
        **kwargs,
    )


def runpod_spec(account_id: str, **kwargs: Any) -> AccountSpec:
    return runpod_account(
        account_id,
        api_key_env=f"IW_RUNPOD_{account_id.upper().replace('-', '_')}_API_KEY",
        **kwargs,
    )


def vast_spec(account_id: str, **kwargs: Any) -> AccountSpec:
    return vast_account(
        account_id,
        api_key_env=f"IW_VAST_{account_id.upper().replace('-', '_')}_API_KEY",
        **kwargs,
    )


def modal_specs(*ids: str) -> list[AccountSpec]:
    return [modal_spec(i) for i in (ids or ("a", "b"))]


def lightning_specs(*ids: str) -> list[AccountSpec]:
    return [lightning_spec(i) for i in (ids or ("a", "b"))]


def runpod_specs(*ids: str) -> list[AccountSpec]:
    return [runpod_spec(i) for i in (ids or ("a", "b"))]


def vast_specs(*ids: str) -> list[AccountSpec]:
    return [vast_spec(i) for i in (ids or ("a", "b"))]


_SPEC_BUILDERS: dict[str, Callable[..., list[AccountSpec]]] = {
    "modal": modal_specs,
    "lightning": lightning_specs,
    "runpod": runpod_specs,
    "vast": vast_specs,
}


def accounts_config(
    providers: Iterable[str] = ("modal",),
    *,
    strategy: str = "round_robin",
    max_attempts: int = 3,
    state_dir: Path | None = None,
    **pool_kwargs: Any,
) -> AccountsConfig:
    """Two sentinel accounts ``a``/``b`` for every provider in ``providers``."""
    return AccountsConfig(
        {
            name: ProviderAccounts(
                accounts=_SPEC_BUILDERS[name](), strategy=strategy, **pool_kwargs
            )
            for name in providers
        },
        max_attempts=max_attempts,
        state_dir=state_dir,
    )


def contains_sentinel(
    text: str | bytes, sentinels: Iterable[str] = SENTINELS
) -> list[str]:
    """Sentinels found in ``text`` (empty list when clean)."""
    if isinstance(text, bytes):
        return [s for s in sentinels if s.encode("utf-8") in text]
    return [s for s in sentinels if s in text]


class FakeClock:
    """Deterministic CredWeave ``Clock``."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._mono = 1000.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    async def sleep_async(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds


# --------------------------------------------------------------------------------------------
# FakeProvider
# --------------------------------------------------------------------------------------------


class Call(NamedTuple):
    op: str
    deployment_id: str | None
    account_id: str


@dataclass
class Step:
    """One scripted outcome of a provider operation."""

    kind: str = "ok"  # ok | fail | raise | hang
    failure: FailureKind | None = None
    retry_after: float | None = None
    resource_may_exist: bool | None = None
    status_code: int | None = None
    message: str = "scripted provider failure"
    exception: BaseException | None = None
    create: bool = False
    """Provision only: the remote resource is created before the failure/hang."""
    exists: bool | None = None
    """resource_exists only: return this value instead of the real state."""


def ok(**kwargs: Any) -> Step:
    return Step("ok", **kwargs)


def fail(
    kind: FailureKind,
    *,
    retry_after: float | None = None,
    resource_may_exist: bool | None = None,
    status_code: int | None = None,
    create: bool = False,
    message: str = "scripted provider failure",
) -> Step:
    return Step(
        "fail",
        failure=kind,
        retry_after=retry_after,
        resource_may_exist=resource_may_exist,
        status_code=status_code,
        create=create,
        message=message,
    )


def crash(exception: BaseException, *, create: bool = False) -> Step:
    """Raise an arbitrary (unclassified) exception, as an SDK would."""
    return Step("raise", exception=exception, create=create)


def hang(*, create: bool = False) -> Step:
    """Block until cancelled (signals ``FakeProvider.entered``)."""
    return Step("hang", create=create)


@dataclass
class FakeProvider(ComputeProvider):
    """Scriptable in-memory cloud.

    ``live`` maps resource name -> creating account id; ``owners`` keeps every resource ever
    created. Script per-operation outcomes with ``script(op, *steps)``; ops are ``provision``,
    ``status``, ``exists`` and ``stop``. Unscripted calls succeed.
    """

    provider_name: str = "modal"
    kind: ProviderType = ProviderType.MODAL
    endpoint_template: str | None = None
    cleanup_failed_deployment: bool = False
    yield_during_provision: bool = True

    calls: list[Call] = field(default_factory=list)
    accounts_seen: list[ProviderAccount] = field(default_factory=list)
    live: dict[str, str] = field(default_factory=dict)
    owners: dict[str, str] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    before_provision: (
        Callable[[DeploymentRecord, ProviderAccount], Awaitable[None]] | None
    ) = None
    _scripts: dict[str, list[Step]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entered = asyncio.Event()

    # --- identity -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.provider_name

    @property
    def provider_type(self) -> ProviderType:
        return self.kind

    def __hash__(self) -> int:  # dataclass(eq=True) would make instances unhashable
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    # --- scripting & inspection ---------------------------------------------------------------

    def script(self, op: str, *steps: Step) -> FakeProvider:
        self._scripts.setdefault(op, []).extend(steps)
        return self

    def _next(self, op: str) -> Step:
        queue = self._scripts.get(op)
        return queue.pop(0) if queue else Step()

    def ops(self, *names: str) -> list[Call]:
        return [c for c in self.calls if not names or c.op in names]

    def accounts_for(self, op: str) -> list[str]:
        return [c.account_id for c in self.calls if c.op == op]

    def adopt(self, resource: str, account_id: str) -> None:
        """Pretends ``resource`` was created earlier under ``account_id``."""
        self.live[resource] = account_id
        self.owners[resource] = account_id

    def endpoint(self, deployment_id: str) -> str:
        if self.endpoint_template:
            return self.endpoint_template.format(id=deployment_id)
        if self.provider_name == "modal":
            return f"https://{deployment_id}.modal.run"
        if self.provider_name == "lightning":
            return f"https://8080-{deployment_id}.cloudspaces.litng.ai"
        return f"http://{deployment_id}.example.test:8080"

    @staticmethod
    def _resource(record: DeploymentRecord) -> str:
        return record.resource.name if record.resource else record.id

    def _record(
        self, op: str, record: DeploymentRecord | None, account: ProviderAccount
    ) -> None:
        self.calls.append(Call(op, record.id if record else None, account.id))
        self.accounts_seen.append(account)

    def _check_owner(
        self, op: str, record: DeploymentRecord, account: ProviderAccount
    ) -> None:
        owner = self.owners.get(self._resource(record))
        if owner is not None and owner != account.id:
            message = (
                f"{op} of {record.id} under account '{account.id}' but it was created by "
                f"'{owner}'"
            )
            self.violations.append(message)
            raise AssertionError(message)

    async def _apply(self, step: Step, op: str) -> None:
        if step.kind == "fail":
            assert step.failure is not None
            raise ProviderOperationError(
                step.message,
                step.failure,
                status_code=step.status_code,
                retry_after=step.retry_after,
                resource_may_exist=step.resource_may_exist,
            )
        if step.kind == "raise":
            assert step.exception is not None
            raise step.exception
        if step.kind == "hang":
            self.entered.set()
            await asyncio.Event().wait()

    # --- ComputeProvider ------------------------------------------------------------------------

    def new_deployment_id(self, profile: ModelProfile) -> str:
        return f"iw-{self.provider_name}-{uuid.uuid4().hex[:10]}"

    def resource_ref(
        self, deployment_id: str, request: DeploymentRequest, account: ProviderAccount
    ) -> ResourceRef:
        return ResourceRef(
            name=deployment_id, scope=account.get_metadata("environment")
        )

    def preflight(
        self,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> None:
        self._record("preflight", None, account)

    def dry_run_endpoint(self, deployment_id: str, runtime: RuntimeSpec) -> str | None:
        return self.endpoint(deployment_id)

    async def provision(
        self,
        record: DeploymentRecord,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
        account: ProviderAccount,
    ) -> ProvisionResult:
        self._record("provision", record, account)
        if self.before_provision is not None:
            await self.before_provision(record, account)
        if self.yield_during_provision:
            await asyncio.sleep(0)
        step = self._next("provision")
        resource = self._resource(record)
        if step.kind == "ok" or step.create:
            self.live[resource] = account.id
            self.owners[resource] = account.id
        try:
            await self._apply(step, "provision")
        finally:
            # This in-memory create cannot outlive the coroutine.
            record.creation_may_continue = False
        return ProvisionResult(
            DeploymentState.STARTING, self.endpoint(record.id), f"rid-{record.id}"
        )

    async def status(
        self, record: DeploymentRecord, account: ProviderAccount
    ) -> ResourceStatus:
        self._record("status", record, account)
        self._check_owner("status", record, account)
        await self._apply(self._next("status"), "status")
        if self._resource(record) in self.live:
            return ResourceStatus(DeploymentState.STARTING, record.endpoint_url)
        return ResourceStatus(DeploymentState.STOPPED, record.endpoint_url)

    async def resource_exists(
        self, record: DeploymentRecord, account: ProviderAccount
    ) -> bool:
        self._record("exists", record, account)
        self._check_owner("exists", record, account)
        step = self._next("exists")
        await self._apply(step, "exists")
        if step.exists is not None:
            return step.exists
        return self._resource(record) in self.live

    async def stop(
        self,
        record: DeploymentRecord,
        account: ProviderAccount,
        action: AutostopAction = AutostopAction.STOP,
    ) -> None:
        self._record("stop", record, account)
        self._check_owner("stop", record, account)
        await self._apply(self._next("stop"), "stop")
        self.live.pop(self._resource(record), None)


# --------------------------------------------------------------------------------------------
# InferWeave builder
# --------------------------------------------------------------------------------------------


def make_weave(
    db_path: Path,
    accounts_config: AccountsConfig | AccountManager | None,
    environ: Mapping[str, str] | None = None,
    providers: Iterable[ComputeProvider] = (),
    *,
    clock: Any | None = None,
    watchdog: MockWatchdogAdapter | None = None,
    probe: MockHealthcheckProbeAdapter | None = None,
    endpoint_auth: EndpointAuthPort | None = None,
    repository: DeploymentRepositoryPort | None = None,
) -> InferWeave:
    """An offline InferWeave "process" over ``db_path`` with ``providers`` registered.

    ``environ`` is referenced (not copied) by the AccountManager, so mutating it simulates
    key rotation or account removal at runtime.
    """
    if isinstance(accounts_config, AccountManager):
        manager = accounts_config
    else:
        manager = AccountManager(
            accounts_config or AccountsConfig(),
            clock=clock,
            environ=environ if environ is not None else {},
        )
    router = ProviderRouter(endpoint_auth=endpoint_auth)
    for provider in providers:
        router.register(provider)
    healthcheck = HealthcheckService(
        probe_port=probe or MockHealthcheckProbeAdapter(default_healthy=True)
    )
    lifecycle = LifecycleService(
        watchdog_port=watchdog or MockWatchdogAdapter(),
        repository=repository or SqliteDeploymentRepository(db_path),
        healthcheck_service=healthcheck,
        provider_resolver=router.get,
        endpoint_auth=endpoint_auth,
        activity_persist_interval_seconds=0.0,
        accounts=manager,
    )
    return InferWeave(
        router=router,
        healthcheck_service=healthcheck,
        lifecycle_service=lifecycle,
        endpoint_auth=endpoint_auth,
        accounts=manager,
    )


async def deploy(weave: InferWeave, provider: str = "modal", **kwargs: Any) -> Any:
    """``weave.deploy`` with test defaults (no readiness wait)."""
    kwargs.setdefault("wait_for_ready", False)
    return await weave.deploy(MODEL, provider=provider, **kwargs)
