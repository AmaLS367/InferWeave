"""Helpers simulating separate InferWeave processes that share one SQLite database."""

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from inferweave import InferWeave
from inferweave.accounts import (
    AccountManager,
    AccountsConfig,
    ProviderAccount,
    ProviderAccounts,
    modal_account,
)
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
from inferweave.clients.transport import InferenceConfig
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.models.deployment import Deployment, DeploymentRequest
from inferweave.models.enums import DeploymentState
from inferweave.models.profile import HealthcheckConfig
from inferweave.ports.auth import EndpointAuthPort
from inferweave.providers.router import ProviderRouter
from inferweave.services.healthcheck_service import HealthcheckService
from inferweave.services.lifecycle_service import LifecycleService

ENDPOINT = "https://iw-test--serve.modal.run"
AUDIO_MODEL = "fish-s2-pro"
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
FAST_RETRIES = InferenceConfig(
    max_retries=3, backoff_base_seconds=0.0, backoff_max_seconds=0.0, jitter_ratio=0.0
)


def make_weave(
    db_path: Path,
    *,
    probe: MockHealthcheckProbeAdapter | None = None,
    endpoint_auth: EndpointAuthPort | None = None,
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
    inference_config: InferenceConfig | None = FAST_RETRIES,
    accounts: AccountsConfig | AccountManager | None = None,
) -> InferWeave:
    """Builds a fresh InferWeave "process" over the SQLite database at ``db_path``."""
    probe = probe or MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck = HealthcheckService(probe_port=probe)
    router = ProviderRouter(endpoint_auth=endpoint_auth)
    account_manager = (
        accounts
        if isinstance(accounts, AccountManager)
        else AccountManager(accounts or AccountsConfig())
    )
    lifecycle = LifecycleService(
        watchdog_port=MockWatchdogAdapter(),
        repository=SqliteDeploymentRepository(db_path),
        healthcheck_service=healthcheck,
        provider_resolver=router.get,
        endpoint_auth=endpoint_auth,
        activity_persist_interval_seconds=0.0,
        accounts=account_manager,
    )
    http_client = (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) if handler else None
    )
    weave = InferWeave(
        router=router,
        healthcheck_service=healthcheck,
        lifecycle_service=lifecycle,
        endpoint_auth=endpoint_auth,
        inference_config=inference_config,
        inference_http_client=http_client,
        accounts=account_manager,
    )
    # Keep readiness polling fast for every built-in model used in tests.
    for model_id in (AUDIO_MODEL, IMAGE_MODEL):
        profile = weave.registry.get(model_id).model_copy(deep=True)
        profile.healthcheck = HealthcheckConfig(
            port=profile.healthcheck.port,
            path=profile.healthcheck.path,
            initial_delay_seconds=0,
            timeout_seconds=0.2,
            probe_interval_seconds=0.01,
        )
        weave.register_model(profile)
    return weave


@contextmanager
def patched_modal(exists: bool = True) -> Any:
    """Patches the Modal SDK edges (deploy, web URL, app lookup/stop) for offline tests.

    ``exists`` is what the (mocked) app lookup reports: ``True`` maps to ``STARTING``,
    ``False`` to ``STOPPED``. Pooled accounts get a mocked ``modal.Client.from_credentials``.
    """
    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value=ENDPOINT),
        patch("modal.Client.from_credentials", return_value=MagicMock()) as from_credentials,
        patch(
            "inferweave.providers.modal_provider.ModalProvider._lookup_app",
            new=AsyncMock(return_value=exists),
        ) as lookup_app,
        patch(
            "inferweave.providers.modal_provider.ModalProvider._stop_modal_app",
            new=AsyncMock(return_value=None),
        ) as stop_app,
    ):
        yield SimpleNamespace(
            lookup_app=lookup_app, stop_app=stop_app, from_credentials=from_credentials
        )


async def deploy_on_modal(
    weave: InferWeave, model: str = AUDIO_MODEL, **kwargs: Any
) -> Deployment:
    """Deploys ``model`` through the real ModalProvider with Modal's network edges mocked."""
    kwargs.setdefault("wait_for_ready", False)
    with patched_modal():
        return await weave.deploy(model=model, provider="modal", **kwargs)


def modal_pool(
    *account_ids: str,
    proxy: bool = True,
    environment: str | None = None,
    environ: dict[str, str] | None = None,
    strategy: str = "round_robin",
    max_attempts: int = 3,
) -> AccountManager:
    """AccountManager with pooled Modal accounts whose secrets are ``SENTINEL-modal-<id>-*``.

    Pass ``environ`` to keep a handle on the (mutable) environment the secrets are read from,
    e.g. to simulate key rotation. Account ``a`` has ``token_id=SENTINEL-modal-a-id``,
    ``token_secret=SENTINEL-modal-a-secret`` and proxy tokens ``SENTINEL-modal-a-proxy-id`` /
    ``SENTINEL-modal-a-proxy-secret``.
    """
    env = environ if environ is not None else {}
    specs = []
    for acct in account_ids:
        key = acct.upper().replace("-", "_")
        env[f"T_{key}_ID"] = f"SENTINEL-modal-{acct}-id"
        env[f"T_{key}_SECRET"] = f"SENTINEL-modal-{acct}-secret"
        if proxy:
            env[f"T_{key}_PID"] = f"SENTINEL-modal-{acct}-proxy-id"
            env[f"T_{key}_PSECRET"] = f"SENTINEL-modal-{acct}-proxy-secret"
        specs.append(
            modal_account(
                acct,
                token_id_env=f"T_{key}_ID",
                token_secret_env=f"T_{key}_SECRET",
                proxy_token_id_env=f"T_{key}_PID" if proxy else None,
                proxy_token_secret_env=f"T_{key}_PSECRET" if proxy else None,
                environment=environment,
            )
        )
    config = AccountsConfig(
        {"modal": ProviderAccounts(accounts=specs, strategy=strategy)},
        max_attempts=max_attempts,
    )
    return AccountManager(config, environ=env)


def modal_record(
    provider: Any,
    profile: Any,
    account: ProviderAccount,
    *,
    state: DeploymentState = DeploymentState.STARTING,
    endpoint_url: str | None = ENDPOINT,
) -> DeploymentRecord:
    """A persisted-style Modal record owned by ``account`` (as the provisioning service writes)."""
    request = DeploymentRequest(model=profile.id, provider="modal")
    deployment_id = provider.new_deployment_id(profile)
    return DeploymentRecord(
        id=deployment_id,
        model=profile.id,
        provider="modal",
        account=account.id,
        resource=provider.resource_ref(deployment_id, request, account),
        state=state,
        endpoint_url=endpoint_url,
    )
