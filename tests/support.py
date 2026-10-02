"""Helpers simulating separate InferWeave processes that share one SQLite database."""

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from inferweave import InferWeave
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
from inferweave.clients.transport import InferenceConfig
from inferweave.models.deployment import Deployment
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
) -> InferWeave:
    """Builds a fresh InferWeave "process" over the SQLite database at ``db_path``."""
    probe = probe or MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck = HealthcheckService(probe_port=probe)
    router = ProviderRouter(endpoint_auth=endpoint_auth)
    lifecycle = LifecycleService(
        watchdog_port=MockWatchdogAdapter(),
        repository=SqliteDeploymentRepository(db_path),
        healthcheck_service=healthcheck,
        provider_resolver=router.get,
        endpoint_auth=endpoint_auth,
        activity_persist_interval_seconds=0.0,
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
def patched_modal(state: DeploymentState = DeploymentState.STARTING) -> Any:
    """Patches the Modal SDK edges (deploy, web URL, app lookup/stop) for offline tests."""
    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value=ENDPOINT),
        patch(
            "inferweave.providers.modal_provider.ModalProvider._fetch_modal_state",
            new=AsyncMock(return_value=state),
        ) as fetch_state,
        patch(
            "inferweave.providers.modal_provider.ModalProvider._stop_modal_app",
            new=AsyncMock(return_value=None),
        ) as stop_app,
    ):
        yield SimpleNamespace(fetch_state=fetch_state, stop_app=stop_app)


async def deploy_on_modal(
    weave: InferWeave, model: str = AUDIO_MODEL, **kwargs: Any
) -> Deployment:
    """Deploys ``model`` through the real ModalProvider with Modal's network edges mocked."""
    kwargs.setdefault("wait_for_ready", False)
    with patched_modal():
        return await weave.deploy(model=model, provider="modal", **kwargs)
